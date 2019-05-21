import candle
import numpy as np
import sdepy
import siglayer
import torch
import torch.nn as nn
import torch.utils.data as torchdata


def gen_data(n_points=100):
    """Generate an Ornstein-Uhlenbeck process."""

    sde = sdepy.ornstein_uhlenbeck_process()
    timeline = np.linspace(0, 1, n_points)
    values = sde(timeline).flatten()
    path = np.c_[timeline, values.tolist()]
    return path.T


def gen_noise(n_points=100):
    """Generate a Brownian motion."""
    
    dt = 1 / np.sqrt(n_points)
    bm = dt * np.r_[0., np.random.randn(n_points - 1).cumsum()]
    timeline = np.linspace(0, 1, n_points)
    return np.c_[timeline, bm].T


def get_signal(num_samples=1000, **kwargs):
    """Generate examples of an Ornstein-Uhlenbeck process."""
    
    paths = np.array([gen_data(**kwargs) for _ in range(num_samples)])
    return torchdata.TensorDataset(torch.tensor(paths, dtype=torch.float))


def get_noise(num_samples=1000, **kwargs):
    """Generate examples of an Brownian motion."""
    
    paths = np.array([gen_noise(**kwargs) for _ in range(num_samples)])
    y = np.zeros_like(paths[:, 0, :-1])
    return torchdata.TensorDataset(torch.tensor(paths, dtype=torch.float), torch.tensor(y, dtype=torch.float))


def scalar_orders(dim, order):
    """The order of the scalar basis elements as one moves along the signature."""
    
    for i in range(order + 1):
        for _ in range(dim ** i):
            yield i


def psi(x, M=4, a=1):
    """Psi function, as defined in the following paper:

    Chevyrev, I. and Oberhauser, H., 2018. Signature moments to
    characterize laws of stochastic processes. arXiv preprint arXiv:1810.10971.

    """

    if x <= M:
        return x
    
    return M + M ** (1 + a) * (M ** (-a) - x ** (-a)) / a
  
    
def normalise_instance(x, order):
    """Normalise signature, following the paper

    Chevyrev, I. and Oberhauser, H., 2018. Signature moments to
    characterize laws of stochastic processes. arXiv preprint arXiv:1810.10971.

    """

    x = torch.cat([torch.tensor([1.], device=x.device), x])

    a = x ** 2
    a[0] -= psi(torch.norm(x))
    
    
    x0 = 1.  # Starting point for Newton-Raphson
    
    moments = torch.tensor([x0 ** (2 * m) for m in range(len(x))], device=x.device)
    polx0 = torch.dot(a, moments)
    
    d_moments = torch.tensor([2 * m * x0 ** (2 * m - 1) for m in range(len(x))], device=x.device)
    d_polx0 = torch.dot(a, d_moments)
    x1 = x0 - polx0 / d_polx0

    if x1 < 0.2:
        x1 = 1.
    
    lambda_ = torch.tensor([x1 ** t for t in scalar_orders(2, order)], device=x.device)

    
    return lambda_ * x


def normalise(x, order):
    """Normalise signature."""

    return torch.stack([normalise_instance(sig, order) for sig in x])


def loss(orig_paths, sig_depth=2, normalise_sigs=True):
    """Loss function is the T statistic defined in
    
    Chevyrev, I. and Oberhauser, H., 2018. Signature moments to
    characterize laws of stochastic processes. arXiv preprint arXiv:1810.10971.

    """
    
    sig = siglayer.Signature(sig_depth)
    orig_signatures = sig(orig_paths)
    if normalise_sigs:
        orig_signatures = normalise(orig_signatures, sig_depth)

    T1 = torch.mean(torch.mm(orig_signatures, orig_signatures.t()))

    def loss_fn(output, *args):
        nonlocal T1, orig_signatures
        T1 = T1.to(device=output.device)
        orig_signatures = orig_signatures.to(device=output.device)
        
        timeline = torch.tensor(np.linspace(0, 1, output.shape[1] + 1), dtype=torch.float32, device=output.device)
        paths = torch.stack([torch.stack([timeline, torch.cat([torch.tensor([0.], device=output.device), path])])
                             for path in output])

        generated_sigs = sig(paths)

        if normalise_sigs:
            generated_sigs = normalise(generated_sigs, sig_depth)

        T2 = torch.mean(torch.mm(orig_signatures, generated_sigs.t()))
        T3 = torch.mean(torch.mm(generated_sigs, generated_sigs.t()))

        return torch.log(T1 - 2 * T2 + T3)

    return loss_fn

    
def create_generative_model():
    return candle.CannedNet((siglayer.Augment((8, 8, 2), 1, include_original=True, include_time=False),
                             candle.Window(2, 0, 1, transformation=siglayer.Signature(3)),
                             siglayer.Augment((1,), 1, include_original=False, include_time=False),
                             candle.batch_flatten  # just squeezing out the channel dimension of length 1
                            ))
