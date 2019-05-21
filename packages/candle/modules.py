import functools as ft
import torch
import torch.nn as nn

from . import utils


class Lambda(nn.Module):
    """Wraps an arbitrary PyTorch function into a Module."""

    def __init__(self, fn, fn_args=(), fn_kwargs=None):
        super(Lambda, self).__init__()
        if isinstance(fn, ft.partial):
            fn.__name__ = fn.func.__name__
            fn.__qualname__ = fn.func.__qualname__
        self.fn = fn
        self.fn_args = fn_args
        self.fn_kwargs = {} if fn_kwargs is None else fn_kwargs

    def forward(self, x):
        return self.fn(x, *self.fn_args, **self.fn_kwargs)
    
    def extra_repr(self):
        return f'fn={self.fn.__qualname__}, fn_args={self.fn_args}, fn_kwargs={self.fn_kwargs}'


class Flatten(nn.Module):
    """Flattening Module."""

    def forward(self, x):
        return utils.batch_flatten(x)


class View(nn.Module):
    """View Module."""

    def __init__(self, shape, **kwargs):
        super(View, self).__init__(**kwargs)
        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)
    
    def extra_repr(self):
        return f'shape={self.shape}'
    

class Concat(nn.Module):
    """Concatenation Module."""

    def __init__(self, dim=-1, **kwargs):
        super(Concat, self).__init__(**kwargs)
        self.dim = dim

    def forward(self, xs):
        return torch.cat(xs, dim=self.dim)
    
    def extra_repr(self):
        return f'dim={self.dim}'


class Split(nn.Module):
    """Split Module."""

    def __init__(self, split, dim=-1, **kwargs):
        super(Split, self).__init__(**kwargs)
        self.split = split
        self.dim = dim

    def forward(self, x):
        return torch.split(x, self.split, self.dim)
    
    def extra_repr(self):
        return f'split={self.split}, dim={self.dim}'


class SkipConnection(nn.Module):
    """Applies a Module, and then adds its input to its output and returns that."""

    def __init__(self, module, **kwargs):
        super(SkipConnection, self).__init__(**kwargs)
        self.module = module

    def forward(self, x):
        y = self.module(x)
        if isinstance(x, (tuple, list)):
            return tuple(xi + yi for xi, yi in zip(x, y))
        else:
            return x + y


class NoInputSpec(nn.Module):
    """Used to create a Module without specifying its number of inputs. A necessary evil is that the Module must
    be called on an example batch of inputs before use, so that it can figure out the input shapes.
    """

    def __init__(self, moduletype, *args, **kwargs):
        self._parameters_property = None
        self._parameters_not_specified = True
        super(NoInputSpec, self).__init__()

        self.moduletype = moduletype
        self.args = args
        self.kwargs = kwargs

        self.module = None

    @property
    def _parameters(self):
        if self._parameters_not_specified:
            raise RuntimeError('Module has not yet been called on an example batch of inputs, so it is not yet '
                               'fully specified.')
        else:
            return self._parameters_property

    @_parameters.setter
    def _parameters(self, item):
        self._parameters_property = item

    def create_module(self, x):
        """The specification for creating a module from the input tensor :x:, and given :args: and :kwarg:."""
        # x.size(1) is the feature dimension of a two-dimensional input, which is used for creating Linear layers, and
        # the channel dimension of a three-dimensional (batch, channel, length) input, which is used to create
        # convolutional layers. Even so this is quite fragile.
        # TODO: Improve this
        return self.moduletype(x.size(1), *self.args, **self.kwargs)

    def forward(self, x):
        if self.module is None:
            # Can't use self.module == None to test because the self.module = self.create_module(x) line actually
            # accesses self._parameters, because it's assigning a Module.
            self._parameters_not_specified = False
            self.module = self.create_module(x)
        return self.module(x)
    
    def extra_repr(self):
        if self.module is None:
            return f'not called yet: moduletype={self.moduletype.__name__}, args={self.args}, kwargs={self.kwargs}'
        else:
            return super(NoInputSpec, self).extra_repr()


class CannedNet(nn.Module):
    """Provides a simple extensible way to specify a neural network without having to define a class. A bit like
    Sequential, but more general.
    - A Module may be specified by something as simple as an integer, for example the width of a Linear layer.
    - In particular the input size of a Linear layer does not need computing in advance
    - The framework may be extended to easily create more complicated nets, for example ResNets; see CannedResNet.

    Subclasses wishing to define how to interpret these 'something more simple' should override the _interpret_element
    method.
    
    As it uses NoInputSpec internally, instances of this Module should be called on an example batch of inputs before
    use.
    """

    def __init__(self, hidden_blocks, debug=False, **kwargs):
        """Create a neural network.

        Note that no activation functions are automatically applied: these should be specified in the :hidden_blocks:
        argument along with everything else.

        Arguments:
            hidden_blocks: A tuple specifying the layers of the network. Subclasses may specify how to interpret the
                values of the tuple. In the default implementation, integers are interpreted as a hidden layer of that
                size, callables are wrapped into a Module, and Modules are used as they are. The documentation of a
                subclass may provide more information on other objects it can interpret. Note that any Module instances
                which are elements of the tuple must have an 'output_shapes' method taking one parameter 'input_shapes',
                specifying the output shapes of the tensors that the module produces, given particular input shapes. (In
                both cases excluding batch dimension.)
            debug: Optional, defaults to False. Whether to print the sizes of Tensors as they go through the network
                layer-by-layer.
        """

        super(CannedNet, self).__init__(**kwargs)

        self.hidden_blocks = hidden_blocks
        self.debug = debug

        self.layers = nn.ModuleList()
        for elem in hidden_blocks:
            self.layers.append(self._interpret_element_wrapper(elem))

    @classmethod
    def spec(cls, *args, **kwargs):
        """Returns a function of no arguments which returns instances of the class with the specified arguments and
        keyword arguments given now.
        """
        def specced():
            return cls(*args, **kwargs)
        return specced

    def __iter__(self):
        # Allows for *unpacking
        return iter(self.layers)

    def _interpret_element(self, elem):
        """Specifies how an element of the :hidden_blocks: argument of __init__ should be interpreted.

        If overriding this method in a subclass, the expected pattern is (note in particular the super() call):

        def _interpret_element(self, current_shapes, elem):
            if isinstance(elem, my_type):
                if is_valid(elem):
                    ...
                    return module
                else:
                    raise ValueError(...)
            return super()._interpret_element(current_shapes, elem)

        Arguments:
            elem: The element of the :hidden_blocks: tuple.

        Returns:
            If :elem: could not be interpreted it will return None.
            If :elem: could be interpreted it will return a module, as specified by :elem:.

        Raises:
            ValueError if :elem: could be interpreted but did not correspond to a well-defined module. For example,
            integers correspond to dense layers of that size. A negative integer would thus raise a ValueError.
        """

        if isinstance(elem, int):
            if elem < 1:
                raise ValueError(f'Integers specifying layers sizes must be greater than or equal to one. Given '
                                 f'{elem}.')
            layer = NoInputSpec(nn.Linear, elem)
            return layer
        elif isinstance(elem, nn.Module):
            return elem
        elif callable(elem):
            return Lambda(elem)

    def _interpret_element_wrapper(self, elem):
        """Wraps the _interpret_element method to check whether or not an element has been interpreted."""

        out = self._interpret_element(elem)
        if out is None:
            raise ValueError(f'Element {elem} of type {type(elem)} in hidden_blocks argument was not understood.')
        return out

    def forward(self, x):
        if self.debug:
            print(f'Input: {x.shape}')
        for layer in self.layers:
            x = layer(x)
            if self.debug:
                print(f'{type(layer).__name__}: {x.shape}')
        return x


class CannedResNet(CannedNet):
    """As CannedNet, but is also capable of understanding tuples as elements of the :hidden_blocks: argument to
    __init__. This tuple element will be interpreted recursively as another CannedResNet, and a skip connection added
     across the layers specified in the tuple.
     """

    def _interpret_element(self, elem):
        if isinstance(elem, (tuple, list)):
            subnet = self.__class__(hidden_blocks=elem)
            return SkipConnection(subnet)
        return super(CannedResNet, self)._interpret_element(elem)
