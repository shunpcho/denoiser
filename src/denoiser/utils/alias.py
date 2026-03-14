from typing import TypedDict

# Shared constants and small aliases used across the denoiser package.
# IMAGE_DIMENSIONS_3D: number of array dimensions for color images (H, W, C)
IMAGE_DIMENSIONS_3D: int = 3

# Default batch entry length when metadata is not included
BATCH_ENTRY_LENGTH_WITHOUT_META: int = 2
BATCH_ENTRY_LENGTH_WITH_META: int = 3

# Pixel and detector constants
MIN_PIXEL_VALUE: int = 0
MAX_PIXEL_VALUE: int = 255
MIN_DETECTOR_COUNT: int = 2


class IndexMapEntry(TypedDict):
    img_i: int
    tile_idx: int
    tile_y: int
    tile_x: int
    tile_h: int
    tile_w: int
    tiles_per_image: int
    orig_h: int
    orig_w: int
    padded_h: int
    padded_w: int
    pad_h: int
    pad_w: int
    image_id: str
