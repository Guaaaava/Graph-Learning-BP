"""Direct FP64 distance for a block of two-dimensional particle coordinates."""
import torch


def distance_2d(left, right):
    """Compute sqrt(dx² + dy²) using the original coordinate differences."""
    if left.shape[-1] != 2 or right.shape[-1] != 2:
        raise ValueError('Expected two-dimensional coordinates')
    if left.dtype != torch.float64 or right.dtype != torch.float64:
        raise ValueError('Expected float64 coordinates')
    dx = left[:, 0, None] - right[None, :, 0]
    dy = left[:, 1, None] - right[None, :, 1]
    dx.square_()
    dy.square_()
    return dx.add_(dy).sqrt_()
