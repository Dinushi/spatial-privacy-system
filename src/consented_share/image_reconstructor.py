from pathlib import Path
from typing import Any, Dict
import cv2
import numpy as np


def paste_bbox_region(
    blurred_image: np.ndarray,
    crop: np.ndarray,
    region_entry: Dict[str, Any],
) -> np.ndarray:
    bbox = region_entry.get("bbox")

    if bbox is None:
        raise ValueError("Cannot reconstruct bbox region because bbox is missing.")

    x1, y1, x2, y2 = map(int, bbox)
    target_w = x2 - x1
    target_h = y2 - y1

    crop_resized = cv2.resize(crop, (target_w, target_h))

    output = blurred_image.copy()
    output[y1:y2, x1:x2] = crop_resized

    return output


def paste_mask_region(
    blurred_image: np.ndarray,
    crop: np.ndarray,
    region_entry: Dict[str, Any],
) -> np.ndarray:
    bbox = region_entry.get("bbox")
    mask_payload = region_entry.get("mask")

    if bbox is None:
        raise ValueError("Cannot reconstruct mask region because bbox is missing.")

    if mask_payload is None:
        raise ValueError("Cannot reconstruct mask region because mask is missing.")

    x1, y1, x2, y2 = map(int, bbox)

    target_w = x2 - x1
    target_h = y2 - y1

    crop_resized = cv2.resize(crop, (target_w, target_h))

    mask = decode_mask_payload(mask_payload)
    mask_uint8 = mask.astype(np.uint8) * 255

    mask_resized = cv2.resize(
        mask_uint8,
        (target_w, target_h),
        interpolation=cv2.INTER_NEAREST,
    )

    mask_bool = mask_resized > 0

    output = blurred_image.copy()
    roi = output[y1:y2, x1:x2]

    roi[mask_bool] = crop_resized[mask_bool]

    output[y1:y2, x1:x2] = roi

    return output


def decode_mask_payload(mask_payload: Dict[str, Any]) -> np.ndarray:
    import base64

    fmt = mask_payload.get("format")
    data = mask_payload.get("data")

    if fmt == "png":
        mask_bytes = base64.b64decode(data.encode("utf-8"))
        mask_arr = np.frombuffer(mask_bytes, dtype=np.uint8)

        mask = cv2.imdecode(mask_arr, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            raise RuntimeError("Failed to decode PNG mask payload.")

        return mask > 0

    raise ValueError(f"Unsupported mask format: {fmt}")

def save_reconstructed_image(
    blurred_image_path: str | Path,
    region_entries_with_crops: list[tuple[Dict[str, Any], np.ndarray]],
    output_path: str | Path,
    placement_mode: str,
) -> str:
    blurred = cv2.imread(str(blurred_image_path))

    if blurred is None:
        raise RuntimeError(f"Failed to read blurred image: {blurred_image_path}")

    reconstructed = blurred.copy()

    for region_entry, crop in region_entries_with_crops:
        if placement_mode == "bbox":
            reconstructed = paste_bbox_region(reconstructed, crop, region_entry)
        elif placement_mode == "mask":
            reconstructed = paste_mask_region(reconstructed, crop, region_entry)
        else:
            raise ValueError(f"Unsupported placement_mode: {placement_mode}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_path), reconstructed)
    return str(output_path)