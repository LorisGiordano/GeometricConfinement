""" IMNPORTS """

import torch
import numpy as np
from scipy.ndimage import binary_fill_holes, label, center_of_mass


""" CENTERS """

def extract_landmarks_from_segmentation(prob_map: torch.Tensor, binary: bool = False):
    
    try:
        B, C, H, W, D = prob_map.shape
    except:
        C, H, W, D = prob_map.shape
        B = 1
        
    centroids = []
    for b in range(B):
        sample_centroids = []
        for c in range(C):
            channel_data = prob_map[b, c].cpu().numpy()
            
            if binary:
                centroid = center_of_mass(channel_data)
            else:
                # binary map of highest probabilities
                threshold = (np.min(channel_data) + np.max(channel_data)) / 2.0
                binary_mask = channel_data > threshold

                # largest component
                labeled, num_features = label(binary_mask)
                if num_features == 0:
                    sample_centroids.append((-1.0,-1.0,-1.0))
                    continue
                sizes = np.bincount(labeled.ravel())
                sizes[0] = 0  # Background is label 0
                largest_label = sizes.argmax()
                largest_component = (labeled == largest_label)

                # compute centroid
                centroid = center_of_mass(channel_data, labels=largest_component)
            sample_centroids.append(centroid)

        centroids.append(torch.tensor(sample_centroids))

    return torch.stack(centroids)


def extract_landmarks_from_heatmaps(heatmaps: torch.Tensor):
    
    try:
        B, C, H, W, D = heatmaps.shape
    except:
        C, H, W, D = heatmaps.shape
        B = 1
        
    centroids = []
    for b in range(B):
        sample_centroids = []
        for c in range(C):
            hmap = heatmaps[b, c]
            max_idx = torch.argmax(hmap)
            x, y, z = torch_unravel_index(max_idx, hmap.shape)
            sample_centroids.append(torch.tensor([x, y, z]))
        centroids.append(torch.stack(sample_centroids))
    return torch.stack(centroids)



def differentiable_centroids(logits: torch.Tensor, eps: float = 1e-6):

    torch.sigmoid(logits)
    
    C, H, W, D = mask.shape
    device = mask.device
    dtype  = mask.dtype

    # Create coordinate grids
    xs = torch.arange(H, device=device, dtype=dtype).view(1, H, 1, 1)
    ys = torch.arange(W, device=device, dtype=dtype).view(1, 1, W, 1)
    zs = torch.arange(D, device=device, dtype=dtype).view(1, 1, 1, D)

    mass = mask.sum(dim=(1,2,3)) + eps        # (C,)
    cx   = (mask * xs).sum(dim=(1,2,3)) / mass
    cy   = (mask * ys).sum(dim=(1,2,3)) / mass
    cz   = (mask * zs).sum(dim=(1,2,3)) / mass

    return torch.stack((cx, cy, cz), dim=1)   # (C,3)



""" """

import itertools
from collections.abc import Sequence
from typing import Union
import operator
from torch.overrides import (
    handle_torch_function,
    has_torch_function,
    has_torch_function_unary,
    has_torch_function_variadic,
)

def torch_unravel_index(indices: torch.Tensor, shape: Union[int, Sequence[int], torch.Size],) -> tuple[torch.tensor, ...]:
    if has_torch_function_unary(indices):
        return handle_torch_function(torch_unravel_index, (indices,), indices, shape=shape)
    res_tensor = _unravel_index(indices, shape)
    return res_tensor.unbind(-1)


def _unravel_index(indices: torch.tensor, shape: Union[int, Sequence[int]]) -> torch.tensor:
    torch._check_type(
        not indices.is_complex()
        and not indices.is_floating_point()
        and not indices.dtype == torch.bool,
        lambda: f"expected 'indices' to be integer dtype, but got {indices.dtype}",
    )

    torch._check_type(
        isinstance(shape, (int, torch.SymInt, Sequence)),
        lambda: f"expected 'shape' to be int or sequence of ints, but got {type(shape)}",
    )

    if isinstance(shape, (int, torch.SymInt)):
        shape = torch.Size([shape])
    else:
        for dim in shape:
            torch._check_type(
                isinstance(dim, (int, torch.SymInt)),
                lambda: f"expected 'shape' sequence to only contain ints, but got {type(dim)}",
            )
        shape = torch.Size(shape)

    torch._check_value(
        all(dim >= 0 for dim in shape),
        lambda: f"'shape' cannot have negative values, but got {tuple(shape)}",
    )

    coefs = list(
        reversed(
            list(
                itertools.accumulate(
                    reversed(shape[1:] + torch.Size([1])), func=operator.mul
                )
            )
        )
    )
    return indices.unsqueeze(-1).floor_divide(
        torch.tensor(coefs, device=indices.device, dtype=torch.int64)
    ) % torch.tensor(shape, device=indices.device, dtype=torch.int64)