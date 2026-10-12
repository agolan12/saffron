"""
Image preparation shared by the training script, shortcut_baseline.py and saliency_map.py.
Kept free of PyTorch so that every script prepares images with exactly the same code.
"""

import numpy as np
from scipy import ndimage
from skimage.filters import threshold_li

# Defaults for li_mask, chosen by eye on 256x256 quadrants at 60x (see the function).
LI_BACKGROUND_SIGMA = 16
LI_MIN_SIZE = 30
LI_MAX_HOLE = 30


def _drop_small_regions(mask, smaller_than):
    """Set to False every connected white region with fewer than `smaller_than` pixels."""
    if smaller_than <= 0:
        return mask
    regions, _ = ndimage.label(mask)
    sizes = np.bincount(regions.ravel())
    keep = sizes >= smaller_than
    keep[0] = False                                    # region 0 is the black part
    return keep[regions]


def li_mask(image, background_sigma=LI_BACKGROUND_SIGMA, min_size=LI_MIN_SIZE, max_hole=LI_MAX_HOLE):
    """
    Black-and-white cell mask of one 2-D image: True = cell, False = background.

    1. Background removal: a blurred copy of the image (Gaussian, width `background_sigma`
       pixels) is subtracted. The blur keeps slow changes such as haze and uneven lighting
       and loses thin structures, so the difference keeps the thin structures.
       0 skips this step.
    2. Li's threshold: one cutoff per image, chosen from that image's own values as the
       best split between a dark group and a bright group of pixels.
    3. Clean-up: white specks smaller than `min_size` pixels are removed, then black holes
       smaller than `max_hole` pixels inside white regions are filled. 0 skips either.

    Same idea as create_microglia_mask and remove_baseline in matt_processing.py, with two
    differences that mattered on these images: the blur is narrower (16 px, not a quarter
    of the image), because a wide blur leaves tissue haze in; and only small holes are
    filled, because filling every hole turns a dense network of branches into one blob.
    Lab recipe for comparison: background_sigma=64, min_size=100, max_hole=65536.
    """
    image = np.asarray(image, dtype=np.float64)
    if background_sigma > 0:
        image = image - ndimage.gaussian_filter(image, background_sigma)
    if image.max() == image.min():
        return np.zeros(image.shape, dtype=bool)       # blank image: nothing stands out
    mask = image > threshold_li(image)
    mask = _drop_small_regions(mask, min_size)
    if max_hole > 0:
        mask = ~_drop_small_regions(~mask, max_hole)
    return mask
