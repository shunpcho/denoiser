from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Callable, Generator


from denoiser.utils.alias import IMAGE_DIMENSIONS_3D


def pad_to_multiple_reflect(
    img: npt.NDArray[np.uint8],
    tile_h: int,
    tile_w: int,
) -> tuple[npt.NDArray[np.uint8], int, int]:
    """Pad right/bottom with reflect so H,W become multiples of tile_h,tile_w.

    Args:
        img: Image array with shape (H, W) or (H, W, C)
        tile_h: Tile height
        tile_w: Tile width

    Returns:
        padded_img: padded image
        pad_h: amount of padding added to bottom
        pad_w: amount of padding added to right
    """
    h, w = img.shape[:2]
    pad_h = (tile_h - (h % tile_h)) % tile_h
    pad_w = (tile_w - (w % tile_w)) % tile_w

    if pad_h == 0 and pad_w == 0:
        return img, 0, 0

    pad_width = ((0, pad_h), (0, pad_w), (0, 0)) if img.ndim == IMAGE_DIMENSIONS_3D else ((0, pad_h), (0, pad_w))

    padded = np.pad(img, pad_width, mode="reflect")
    return padded.astype(img.dtype, copy=False), pad_h, pad_w


def iter_image_tiles(
    tile_size: int | tuple[int, int],
) -> Callable[[npt.NDArray[np.uint8 | np.float32]], Generator[npt.NDArray[np.uint8 | np.float32], None, None]]:
    """Create function to iterate (yield) tiles from an image by grid.

    Notes:
        - This function does NOT pad. If you want full coverage with fixed tile size,
          pad the image to multiples of tile size beforehand.

    Args:
        tile_size: If int, square tile. If tuple, (tile_h, tile_w).

    Returns:
        A function that takes an image (H, W[, C]) and yields tiles in row-major order.
    """
    if isinstance(tile_size, int):
        tile_h = tile_w = tile_size
    else:
        tile_h, tile_w = tile_size

    def _iter(
        img: npt.NDArray[np.uint8 | np.float32],
    ) -> Generator[npt.NDArray[np.uint8 | np.float32], None, None]:
        h, w = img.shape[:2]

        for y in range(0, h, tile_h):
            for x in range(0, w, tile_w):
                if img.ndim == IMAGE_DIMENSIONS_3D:
                    yield img[y : y + tile_h, x : x + tile_w, :]
                else:
                    yield img[y : y + tile_h, x : x + tile_w]

    return _iter
