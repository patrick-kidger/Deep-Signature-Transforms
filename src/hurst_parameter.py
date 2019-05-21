import candle
import fbm
import iisignature
import numpy as np
import random
import siglayer
import sklearn.base as base
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torchdata
import torchvision.transforms as transforms


class AddTime(base.BaseEstimator, base.TransformerMixin):
    """Augments the path with time."""
    def __init__(self, init_time=0.):
        self.init_time = init_time

    def fit(self, X, y=None):
        return self

    def transform_instance(self, X):
        t = np.linspace(self.init_time, self.init_time + 1, len(X))
        return np.c_[t, X]

    def transform(self, X, y=None):
        return [self.transform_instance(x) for x in X]

    
def generate_fBM(n_paths, n_samples, hurst_exponents):
    """Generate FBM paths"""
    X = []
    y = []
    for j in range(n_paths):
        hurst = random.choice(hurst_exponents)
        X.append(fbm.FBM(n=n_samples, hurst=hurst, length=1, method='daviesharte').fbm())
        y.append(hurst)
    return np.array(X), np.array(y)


def generate_data(n_paths_train, n_paths_test, n_samples, hurst_exponents):
    """Generate train and test datasets"""
   
    # generate dataset
    x_train, y_train = generate_fBM(n_paths_train, n_samples, hurst_exponents)
    x_test, y_test = generate_fBM(n_paths_test, n_samples, hurst_exponents)

    # reshape targets
    y_train = np.expand_dims(y_train, axis=1)
    y_test = np.expand_dims(y_test, axis=1)
    
    return x_train, y_train, x_test, y_test


def preprocess_data(x_train, x_test, flag=None):
    """Peforms model-dependent preprocessing."""
    if flag == 'neuralsig':
        # We don't need to backprop through the signature if we're just building a model on top
        # so we actually perform the signature here as a feature transformation, rather than in
        # the model.
        path_transform = AddTime()
        x_train = np.array([iisignature.sig(x, 4) for x in path_transform.fit_transform(x_train)])
        x_test = np.array([iisignature.sig(x, 4) for x in path_transform.fit_transform(x_test)])
    elif flag == 'lstm':
        # LSTM wants another dimension in one place...
        x_train = np.expand_dims(x_train, 2)
        x_test = np.expand_dims(x_test, 2)
    else:
        # ...everyone else wants the extra dimension in another
        x_train = np.expand_dims(x_train, 1)
        x_test = np.expand_dims(x_test, 1)
    return x_train, x_test


def generate_torch_batched_data(x_train, y_train, x_test, y_test, train_batch_size, test_batch_size):
    """Generate torch dataloaders"""

    # make torch dataset
    train_dataset = torchdata.TensorDataset(torch.tensor(x_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.float))
    test_dataset = torchdata.TensorDataset(torch.tensor(x_test, dtype=torch.float), torch.tensor(y_test, dtype=torch.float))

    # process with torch dataloader
    train_dataloader = torchdata.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=8)
    test_dataloader = torchdata.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=8)

    example_batch_x, example_batch_y = next(iter(train_dataloader))

    return train_dataloader, test_dataloader, example_batch_x, example_batch_y


def hurst_rescaled_range(ts):
    """Uses the rescaled range method to estimate the Hurst parameter."""
    
    # calculate standard deviation of differenced series using various lags
    lags = range(2, 20)
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags] 
    # calculate Hurst as slope of log-log plot
    m = np.polyfit(np.log(lags), np.log(tau), 1)
    hurst = m[0]*2.0
    return hurst


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, final_nonlinearity=lambda x: x, **kwargs):
        super(LSTM, self).__init__(**kwargs)

        self.mod = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.final = final_nonlinearity

    def forward(self, x):
        out, _ = self.mod(x)
        out = out[:, -1, :]
        return self.final(self.fc(out))


class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, final_nonlinearity=lambda x: x, **kwargs):
        super(GRU, self).__init__(**kwargs)
        
        self.mod = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.final = final_nonlinearity

    def forward(self, x):
        out, _ = self.mod(x)
        out = out[:, -1, :]
        return self.final(self.fc(out))

        
# Now THIS is deep signatures!
def deep_recurrent(output_shape, sig=True, sig_depth=4, final_nonlinearity=lambda x: x,
                   augment_layer_sizes=(32, 32, 2), augment_kernel_size=8, augment_include_original=True,
                   augment_include_time=True,
                   lengths=(5, 5, 10), strides=(1, 1, 5), adjust_lengths=(0, 0, 0), memory_sizes=(8, 8, 8),
                   layer_sizes_s=((32,), (32,), (32, 16)), hidden_output_sizes=(8, 8)):
    """This model stacks multiple layers of signatures on top of one another in a natural way.
    
    - Augment the features with something learnable
    - Slide a window across the augmented features
    - Take the signature of each window
    - Put this list of signatures back together to recover the path dimension
    - Apply an RNN across the path dimension, preserving the intermediate outputs, so the path dimension is preserved
    - Slide another window
    - Take another signature
    - Reassemble signatures along path dimension
    - Another RNN
    - ... 
    - etc. for some number of times
    - ...
    - Slide another window
    - Take another signature
    - Reassemble signatures along path dimension
    - Another RNN; this time throw away intermediate outputs and just present the final output as the overall output.
    If :sig: is falsy then the signature layers will be replaced with flattening instead.
    It expects input tensors of three dimensions: (batch, channels, length).
    
    For a simpler example in the same vein, see siglayer.examples.create_windowed.

    Arguments:
        output_shape: The final output shape from the network.
        sig: Optional, whether to use signatures in the network. If True a signature will be applied between each 
            window. If False then the output is simply flattened. Defaults to True.
        sig_depth: Optional. If signatures are used, then this specifies how deep they should be truncated to.
        final_nonlinearity: Optional. What final nonlinearity to feed the final tensors of the network through, e.g. a
            sigmoid when desiring output between 0 and 1. Defaults to the identity.
        augment_layer_sizes: Optional. A tuple of integers specifying the size of the hidden layers of the feedforward
            network that is swept across the input stream to augment it. May be set to the empty tuple to do no 
            augmentation.
        augment_kernel_size: Optional. How far into the past the swept feedforward network (that is doing augmenting) 
            should take inputs from. For example if this is 1 then it will just take data from a single 'time', making
            it operate in a 'pointwise' manner. If this is 2 then it will take the present and the most recent piece of
            past information, and so on.
        augment_include_original: Optional. Whether to include the original path in the augmentation.
        augment_include_time: Optional. Whether to include an increasing 'time' parameter in the augmentation.
        lengths, strides, adjust_lengths, memory_sizes: Optional. Should each be a tuple of integers, all of the same
            length as one another. The length of these arguments determines the number of windows; this length must be
            at least one. The ith values determine the length, stride and adjust_length arguments of the ith Window,
            and the size of the memory of the ith RNN.
        layer_sizes_s: Optional. Should be a tuple of the same length as lengths, strides, adjust_lengths, 
            memory_sizes. Each element of the tuple should itself be a tuple of integers specifying the sizes of the
            hidden layers of each RNN.
        hidden_output_sizes: Optional. Should be a tuple of integers one shorter than the length of lengths, strides,
            adjust_lengths, memory_sizes. It determines the output size of each RNN. It is of a slightly shorter length
            because the final output size is actually already determined by the output_shape argument!
    """
    
    num_windows = len(lengths)
    assert num_windows >= 1
    assert len(strides) == num_windows
    assert len(adjust_lengths) == num_windows
    assert len(layer_sizes_s) == num_windows
    assert len(memory_sizes) == num_windows
    assert len(hidden_output_sizes) == num_windows - 1

    if sig:
        transformation = siglayer.Signature(depth=sig_depth)
    else:
        transformation = lambda x: candle.batch_flatten(x.contiguous())
        
    final_output_size = torch.Size(output_shape).numel()
    output_sizes = (*hidden_output_sizes, final_output_size)
    
    recurrent_layers = []
    for (i, length, stride, adjust_length, layer_sizes, memory_size, output_size
         ) in zip(range(num_windows), lengths, strides, adjust_lengths, layer_sizes_s, memory_sizes, output_sizes):
        
        window_layers = []
        for layer_size in layer_sizes:
            window_layers.append(layer_size)
            window_layers.append(F.relu)
            
        intermediate_outputs = (num_windows - 1 != i)

        recurrent_layers.append(candle.Window(length=length, stride=stride, adjust_length=adjust_length,
                                              transformation=transformation))
        recurrent_layers.append(candle.Recur(module=candle.CannedNet((candle.Concat(),
                                                                      *window_layers,
                                                                      memory_size + output_size,
                                                                      candle.Split((memory_size, output_size)))),
                                             memory_shape=(memory_size,),
                                             intermediate_outputs=intermediate_outputs))

    return candle.CannedNet((siglayer.Augment(layer_sizes=augment_layer_sizes, 
                                              kernel_size=augment_kernel_size,
                                              include_original=augment_include_original, 
                                              include_time=augment_include_time),
                             *recurrent_layers,
                             candle.View(output_shape),
                             final_nonlinearity))
