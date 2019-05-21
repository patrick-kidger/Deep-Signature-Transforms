import torch


def convert_to_tensor(tensor, **kwargs):
    """Converts the argument :tensor: to a tensor if it isn't already one. If converted to a tensor, then any :**kwargs:
    will be passed as well.

    This is useful (over torch.tensor or torch.as_tensor) when using torch.jit.trace, which interprets both torch.tensor
    and torch.as_tensor as static values; this will correctly use the same tensor that was passed, if possible.
    """
    if isinstance(tensor, torch.Tensor):
        return tensor
    else:
        return torch.tensor(tensor, **kwargs)


# For tracing, when everything gets turned into a tensor.
Integer = (int, torch.IntTensor, torch.LongTensor)
