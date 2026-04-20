from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


class CropExtractor:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_bbox_crop(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]

        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            return np.empty((0, 0, 3), dtype=frame.dtype)

        return frame[y1:y2, x1:x2].copy()

    def extract_mask_crop(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> np.ndarray:
        if bbox is None:
            ys, xs = np.where(mask > 0)
            if len(xs) == 0 or len(ys) == 0:
                return np.empty((0, 0, 3), dtype=frame.dtype)
            x1, x2 = int(xs.min()), int(xs.max()) + 1
            y1, y2 = int(ys.min()), int(ys.max()) + 1
        else:
            x1, y1, x2, y2 = bbox

        crop = frame[y1:y2, x1:x2].copy()
        local_mask = mask[y1:y2, x1:x2]

        if crop.size == 0:
            return crop

        local_mask_3 = np.repeat((local_mask > 0)[:, :, None], 3, axis=2)
        crop = np.where(local_mask_3, crop, 0)
        return crop

    def save_crop(
        self,
        crop: np.ndarray,
        frame_idx: int,
        object_idx: int,
        label: str,
    ) -> Optional[str]:
        if crop.size == 0:
            return None

        safe_label = label.replace(" ", "_").replace("/", "_")
        filename = f"frame_{frame_idx:06d}_obj_{object_idx:02d}_{safe_label}.png"
        path = self.output_dir / filename
        cv2.imwrite(str(path), crop)
        return str(path)