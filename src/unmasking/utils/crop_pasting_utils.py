from __future__ import annotations

from typing import Any, Dict
import cv2
import numpy as np


def decode_png_crop(png_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(png_bytes, dtype=np.uint8)
    crop = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if crop is None:
        raise RuntimeError("Failed to decode decrypted PNG crop.")

    return crop


def paste_bbox_crop(
    frame: np.ndarray,
    crop: np.ndarray,
    bbox: list[int],
) -> np.ndarray:
    x1, y1, x2, y2 = map(int, bbox)

    target_w = x2 - x1
    target_h = y2 - y1

    if crop.shape[0] != target_h or crop.shape[1] != target_w:
        crop = cv2.resize(crop, (target_w, target_h))

    frame[y1:y2, x1:x2] = crop
    return frame


def paste_mask_crop(
    frame: np.ndarray,
    crop: np.ndarray,
    region_entry: Dict[str, Any],
) -> np.ndarray:
    bbox = region_entry["bbox"]
    x1, y1, x2, y2 = map(int, bbox)

    target_w = x2 - x1
    target_h = y2 - y1

    if crop.shape[0] != target_h or crop.shape[1] != target_w:
        crop = cv2.resize(crop, (target_w, target_h))

    mask_info = region_entry.get("mask")

    if not mask_info:
        return paste_bbox_crop(frame, crop, bbox)

    mask_b64 = mask_info["data"]
    import base64

    mask_bytes = base64.b64decode(mask_b64.encode("utf-8"))
    mask_arr = np.frombuffer(mask_bytes, dtype=np.uint8)
    mask = cv2.imdecode(mask_arr, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        return paste_bbox_crop(frame, crop, bbox)

    if mask.shape[0] != target_h or mask.shape[1] != target_w:
        mask = cv2.resize(mask, (target_w, target_h))

    mask_bool = mask > 0

    roi = frame[y1:y2, x1:x2]
    roi[mask_bool] = crop[mask_bool]
    frame[y1:y2, x1:x2] = roi

    return frame