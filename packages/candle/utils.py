import torch


def batch_fn(fn):
    """Transforms a function :fn: to act on each element of a batch individually."""
    def batched_fn(x, **kwargs):
        return torch.stack([fn(xi, **kwargs) for xi in x])
    return batched_fn


def outer_product(tensor1, tensor2):
    """Computes the outer product of two tensors."""
    return torch.tensordot(tensor1.unsqueeze(0), tensor2.unsqueeze(0), dims=((0,), (0,)))


def flatten(tensor):
    """Flattens a tensor."""
    return tensor.view(-1)


def batch_flatten(tensor):
    """Flattens a tensor except for the batch dimension."""
    return tensor.view(tensor.size(0), -1)


def cat(tensors, dim=0, out=None):
    """As torch.cat, but returns the original tensor if len(tensors) == 1, so that an unneeded copy is not made."""
    if len(tensors) == 1:
        return tensors[0]
    else:
        return torch.cat(tensors, dim=dim, out=out)


def stack(tensors, dim=0, out=None):
    """As torch.stack, but returns the original tensor if len(tensors) == 1, so that an unneeded copy is not made."""
    if len(tensors) == 1:
        return tensors[0].unsqueeze(dim=dim)
    else:
        return torch.stack(tensors, dim=dim, out=out)
