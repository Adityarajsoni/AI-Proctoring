# utils/image_preprocess.py
"""Image preprocessing utilities — exact mirror of fas_test_repo/utils.py."""

import cv2
import numpy as np
import torch


def xyxy2xywh(bbox: np.ndarray | list) -> np.ndarray:
    """Convert bounding box from [x1, y1, x2, y2] to [x, y, w, h] format."""
    if isinstance(bbox, list):
        bbox = np.array(bbox)

    result = np.copy(bbox)
    result[..., 2] = bbox[..., 2] - bbox[..., 0]
    result[..., 3] = bbox[..., 3] - bbox[..., 1]
    return result


def crop_face(
    image: np.ndarray,
    bbox: list[int],
    scale: float,
    out_w: int,
    out_h: int,
) -> np.ndarray:
    """Crop and resize face region using scaled expansion around face center.

    This is the EXACT crop_face from fas_test_repo/utils.py.
    The scale factor expands the crop window so the model receives
    the spatial context (forehead, neck) it was trained on.

    Args:
        image: Input BGR image.
        bbox:  Bounding box in [x, y, w, h] format.
        scale: Expansion factor (2.7 for MiniFASNetV2, 4.0 for V1SE).
        out_w: Output width (80).
        out_h: Output height (80).

    Returns:
        Cropped and resized face patch (out_h × out_w, BGR).
    """
    src_h, src_w = image.shape[:2]
    x, y, box_w, box_h = bbox

    scale = min((src_h - 1) / box_h, (src_w - 1) / box_w, scale)
    new_w = box_w * scale
    new_h = box_h * scale

    center_x = x + box_w / 2
    center_y = y + box_h / 2

    x1 = max(0, int(center_x - new_w / 2))
    y1 = max(0, int(center_y - new_h / 2))
    x2 = min(src_w - 1, int(center_x + new_w / 2))
    y2 = min(src_h - 1, int(center_y + new_h / 2))

    cropped = image[y1 : y2 + 1, x1 : x2 + 1]
    return cv2.resize(cropped, (out_w, out_h))


def to_tensor(image: np.ndarray) -> torch.Tensor:
    """Convert numpy HWC image to CHW float tensor.

    Exact mirror of fas_test_repo/utils.py::to_tensor().
    No normalisation is applied — the model was trained without it.
    """
    if image.ndim == 2:
        image = image[:, :, np.newaxis]

    return torch.from_numpy(image.transpose(2, 0, 1)).float()