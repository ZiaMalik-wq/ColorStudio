import cv2
import numpy as np

# Default color palette for slices (BGR)
DEFAULT_COLORS = [
    (139, 0,   0  ),   # dark blue
    (255, 100, 0  ),   # blue-orange
    (0,   200, 50 ),   # green
    (0,   220, 220),   # yellow
    (0,   140, 255),   # orange
    (0,   0,   220),   # red
    (200, 0,   180),   # purple
    (255, 255, 255),   # white
]

def apply_level_slicing(
    gray_img:   np.ndarray,
    n_slices:   int         = 6,
    colors:     list        = None,
    background: bool        = False,
) -> np.ndarray:
    """
    Divide intensity range [0, 255] into n_slices equal bands
    and paint each band a distinct color.

    Args:
        gray_img:    Grayscale image (H, W) or (H, W, 1)
        n_slices:    Number of intensity bands (2-8)
        colors:      List of BGR tuples, one per slice. Defaults to DEFAULT_COLORS.
        background:  If True, pixels outside defined slices stay gray.
                     If False, all pixels get a color.

    Returns:
        colored:     BGR image (H, W, 3) uint8
    """
    gray = _prepare_gray(gray_img)
    n_slices = max(2, min(8, n_slices))

    if colors is None:
        colors = DEFAULT_COLORS[:n_slices]
    else:
        colors = colors[:n_slices]
        # pad if fewer colors than slices
        while len(colors) < n_slices:
            colors.append(DEFAULT_COLORS[len(colors) % len(DEFAULT_COLORS)])

    # Create a 256-element LUT
    # Default: map everything to its grayscale equivalent
    lut = np.stack([np.arange(256)] * 3, axis=-1).astype(np.uint8)
    
    if background:
        # Preserve background by starting slices at intensity 30
        boundaries = np.linspace(30, 255, n_slices + 1, dtype=int)
    else:
        boundaries = np.linspace(0, 255, n_slices + 1, dtype=int)
        
    for i in range(n_slices):
        low, high = int(boundaries[i]), int(boundaries[i + 1])
        # To avoid overlaps and ensure full coverage up to the boundaries
        if i == n_slices - 1:
            lut[low:high+1] = colors[i]
        else:
            lut[low:high] = colors[i]

    lut = lut.reshape(256, 1, 3)
    colored = cv2.LUT(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), lut)
    return colored


def apply_custom_slicing(
    gray_img:    np.ndarray,
    thresholds:  list,    # e.g. [50, 120, 200]
    colors:      list,    # e.g. [(255,0,0), (0,255,0), (0,0,255), (255,255,0)]
) -> np.ndarray:
    """
    Apply level slicing with user-defined threshold values.

    Args:
        gray_img:    Grayscale image
        thresholds:  List of N intensity boundaries (0-255), sorted ascending
        colors:      List of N+1 BGR colors — one per region between thresholds

    Returns:
        colored:     BGR image (H, W, 3) uint8
    """
    gray = _prepare_gray(gray_img)
    lut = np.stack([np.arange(256)] * 3, axis=-1).astype(np.uint8)

    thresholds = sorted(thresholds)
    boundaries = [0] + thresholds + [255]

    for i in range(len(boundaries) - 1):
        low, high = boundaries[i], boundaries[i + 1]
        color = colors[i] if i < len(colors) else (255, 255, 255)
        if i == len(boundaries) - 2:
            lut[low:high+1] = color
        else:
            lut[low:high] = color

    lut = lut.reshape(256, 1, 3)
    colored = cv2.LUT(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), lut)
    return colored


def _prepare_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = img[:, :, 0]
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8) \
              if img.max() <= 1.0 \
              else np.clip(img, 0, 255).astype(np.uint8)
    return img