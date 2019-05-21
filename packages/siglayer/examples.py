import candle
import torch
import torch.nn.functional as F

from . import backend
from . import modules


def create_feedforward(output_shape, sig=True, sig_depth=4, final_nonlinearity=lambda x: x, 
                       layer_sizes=(32, 32, 32)):
    """This simple model uses a few hidden layers with signature nonlinearities between them.
    If :sig: is falsy then the signature layers will be replaced with ReLU instead.
    It expects input tensors of two dimensions: (batch, features).

    Note that whilst this is a simple example, this is fundamentally quite a strange idea (with sig=True).
    There's no natural path-like structure here.
    """

    if sig:
        nonlinearity = lambda: modules.ViewSignature(channels=2, length=16, sig_depth=sig_depth)
    else:
        nonlinearity = lambda: F.relu

    layers = []
    for layer_size in layer_sizes:
        layers.append(layer_size)
        layers.append(nonlinearity())
    return candle.CannedNet((candle.Flatten(),
                             *layers,
                             torch.Size(output_shape).numel(),
                             candle.View(output_shape),
                             final_nonlinearity))


def create_simple(output_shape, sig=True, sig_depth=4, final_nonlinearity=lambda x: x, 
                  augment_layer_sizes=(8, 8, 2), augment_kernel_size=1, 
                  augment_include_original=True,  augment_include_time=True, 
                  layer_sizes=(32, 32)):
    """This model uses a single signature layer:
        - Augment the features with something learnable
        - Apply signature
        - Small ReLU network afterwards.
    If :sig: is falsy then the signature layers will be replaced with flatten-and-ReLU instead.
    It expects input tensors of three dimensions: (batch, channels, length).
    """

    if sig:
        siglayer = (backend.Signature(sig_depth),)
    else:
        siglayer = (candle.Flatten(), F.relu)

    layers = []
    for layer_size in layer_sizes:
        layers.append(layer_size)
        layers.append(F.relu)

    return candle.CannedNet((modules.Augment(layer_sizes=augment_layer_sizes,
                                             kernel_size=augment_kernel_size,
                                             include_original=augment_include_original,
                                             include_time=augment_include_time),
                             *siglayer,
                             *layers,
                             torch.Size(output_shape).numel(),
                             candle.View(output_shape),
                             final_nonlinearity))


# This example is deliberately not as flexible as the other examples here, to make it easier
# to understand
def create_windowed(output_shape, sig_depth=4, final_nonlinearity=lambda x: x):
    """This model applies two signature layers:
        - Augment the features with something learnable
        - Apply signature
        - Recurrent network
        - Apply signature
        - Recurrent network
        - take the final output of the last recurrent block and reshape and return it.
    
    Basically it applies a couple of RNNs to the input data with signatures in between them.
    """
    
    if sig:
        transformation = lambda: backend.Signature(depth=sig_depth)
    else:
        transformation = lambda: candle.batch_flatten
        
    output_size = torch.Size(output_shape).numel()

    return candle.CannedNet((modules.Augment(layer_sizes=(16, 16, 2), kernel_size=4),
                             candle.Window(length=5, stride=1, transformation=backend.Signature(depth=sig_depth)),
                             # We could equally well have an Augment here instead of a Recur; both are path-preserving
                             # neural networks.
                             candle.Recur(module=candle.CannedNet((candle.Concat(),
                                                                   32, F.relu,
                                                                   16,  # memory size + output size
                                                                   candle.Split((8, 8)))),  # memory size, output size
                                          memory_shape=(8,)),  # memory size
                             candle.Window(length=10, stride=5, transformation=backend.Signature(depth=sig_depth)),
                             candle.Recur(module=candle.CannedNet((candle.Concat(),
                                                                   32, F.relu, 16, F.relu,
                                                                   8 + output_size,  # memory size + output size
                                                                   candle.Split((8, output_size)))),  # memory size, output size
                                          memory_shape=(8,),  # memory size
                                          intermediate_outputs=False),
                             candle.View(output_shape),
                             final_nonlinearity))
