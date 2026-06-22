from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from common.security import aes_gcm_encrypt, b64e

# this class has utils required to encrypt and pack a given privacy region

@dataclass
class PrivateRegionEntryInput:
    region_id: str
    object_id: str
    object_privacy_category: str
    frame_idx: int
    bbox: tuple[int, int, int, int] | None
    crop: np.ndarray
    placement_mode: str
    mask: Optional[np.ndarray] = None


def encode_png_bytes(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Failed to PNG-encode crop or mask.")
    return bytes(buf)


def encode_mask_payload(mask: np.ndarray | None, bbox: tuple[int, int, int, int] | None) -> Dict[str, Any] | None:
    if mask is None or bbox is None:
        return None

    x1, y1, x2, y2 = bbox
    local_mask = mask[y1:y2, x1:x2]
    local_mask_u8 = (local_mask.astype(np.uint8) * 255) if local_mask.dtype != np.uint8 else local_mask
    mask_png = encode_png_bytes(local_mask_u8)

    return {
        "format": "png",
        "width": int(local_mask_u8.shape[1]),
        "height": int(local_mask_u8.shape[0]),
        "data": b64e(mask_png),
    }


def build_private_region_entry(inp: PrivateRegionEntryInput, object_key: bytes) -> Dict[str, Any]:
    crop_png = encode_png_bytes(inp.crop)
    enc = aes_gcm_encrypt(object_key, crop_png)

    entry: Dict[str, Any] = {
        "region_id": inp.region_id,
        "object_id": inp.object_id,
        "object_privacy_category": inp.object_privacy_category,
        "frame_idx": int(inp.frame_idx),
        "bbox": list(inp.bbox) if inp.bbox is not None else None,
        "crop_shape": list(inp.crop.shape),
        "encoding": "png",
        "mask": encode_mask_payload(inp.mask, inp.bbox) if inp.placement_mode == "mask" else None,
        "crypto": {
            "alg": enc["alg"],
            "nonce": enc["nonce"],
            "tag": enc["tag"],
        },
        "ciphertext": enc["ciphertext"],
    }
    return entry