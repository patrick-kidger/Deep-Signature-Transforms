import candle
import iisignature
import torch
import torch.autograd as autograd
import torch.nn as nn
import warnings


def sig_dim(alphabet_size, depth):
    """Calculates the number of terms in a signature of depth :depth: over an alphabet of size :alphabet_size:."""
    return int(alphabet_size * (1 - alphabet_size ** depth) / (1 - alphabet_size))
    # == sum(alphabet_size ** i for i in range(1, depth + 1)) (geometric sum formula)


class path_sig_fn(autograd.Function):
    """An autograd.Function corresponding to the signature map. See also siglayer/backend/pytorch_implementation.py."""

    @staticmethod
    def forward(ctx, path, depth):
        device = path.device
        # transpose because the PyTorch convention for convolutions is channels first. The iisignature expectation is
        # that channels are last.
        path = path.detach().cpu().numpy().transpose()  # sloooow CPU :(
        ctx.path = path
        ctx.depth = depth
        return torch.tensor(iisignature.sig(path, depth), dtype=torch.float, device=device)

    @staticmethod
    def backward(ctx, grad_output):
        device = grad_output.device
        backprop = iisignature.sigbackprop(grad_output.cpu().numpy(), ctx.path, ctx.depth)
        # transpose again to go back to the PyTorch convention of channels first
        out = torch.tensor(backprop, dtype=torch.float, device=device).t()

        # better safe than sorry
        # https://discuss.pytorch.org/t/when-should-you-save-for-backward-vs-storing-in-ctx/6522/9
        # not sure this is actually necessary though
        del ctx.path
        del ctx.depth
        return out, None


def path_sig(path, depth):
    """Calculates the signature transform of a :path: to signature depth :depth:."""
    return path_sig_fn.apply(path, depth)
    
    
batch_path_sig = candle.batch_fn(path_sig)


class Signature(nn.Module):
    """Given some path mapping from, say, [0, 1] into \reals^d, we may define the 'signature' of the path as a
    particular sigtensor with respect to an alphabet of n letters. (Note how d is the target dimension of the path.)
    That is, the signature is a map from the space of paths to the tensor algebra. Up to certain mathematical niceties,
    this map may be inverted; the signature is sufficient to define the path. (Technically speaking, it defines the path
    up to 'tree-like equivalence': this means that the signature does not pick up on back-tracking)

    Thus the signature is a natural way to characterise a path; in the language of machine learning is an excellent
    feature map.

    Given a tensor of shape (x, y), then one may interpret this a piecewise constant path from [0, x] into \reals^y,
    changing its value at each integer. Whether this is a natural interpretation depends on the data that the tensor
    represents, of course, but this allows for taking the signature of a tensor, which is precisely what this Module
    does.
    """

    def __init__(self, depth, **kwargs):
        if not isinstance(depth, candle.Integer) or depth < 1:
            raise ValueError(f'Depth must be an integer greater than or equal to one. Given {depth} of type '
                             f'{type(depth)}')
        super(Signature, self).__init__(**kwargs)
        self.depth = depth

    def forward(self, path):
        if path.size(1) == 1:
            warnings.warn(f'{self.__class__.__name__} called on path with only one channel; the signature is now just '
                          f'the moments of the path, so there is no interesting information from cross terms.')
        # path is expected to be a 3-dimensional tensor, with batch, channel and length axes respectively, say of shape
        # (b, c, l). Each batch element is treated separately. Then values are interpreted as l sample points from a
        # path in \reals^c
        return batch_path_sig(path, depth=self.depth)

    def extra_repr(self):
        return f'depth={self.depth}'
