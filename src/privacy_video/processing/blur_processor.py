from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


class BlurProcessor:
    def __init__(self, ksize: Tuple[int, int] = (201, 201)) -> None:
        self.ksize = ksize

    def blur_bbox(
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
            return frame

        roi = frame[y1:y2, x1:x2]
        blurred = cv2.GaussianBlur(roi, self.ksize, 0)
        frame[y1:y2, x1:x2] = blurred
        return frame

    def blur_mask(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        mask: binary or bool mask with shape [H, W]
        """
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)

        blurred_full = cv2.GaussianBlur(frame, self.ksize, 0)

        mask_3 = np.repeat(mask[:, :, None], 3, axis=2)
        frame = np.where(mask_3 > 0, blurred_full, frame)
        return frame

    def process(
        self,
        frame: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if mask is not None:
            return self.blur_mask(frame, mask)
        if bbox is not None:
            return self.blur_bbox(frame, bbox)
        return frame