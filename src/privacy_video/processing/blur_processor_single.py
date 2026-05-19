from __future__ import annotations

from typing import List, Optional, Tuple
from .blur_processor import BlurProcessor

import cv2
import numpy as np


class CombinedMaskBlurProcessor(BlurProcessor):
    """
    Child blur processor that applies blur once using a combined mask
    from multiple object masks.
    """

    def blur_combined_mask(
        self,
        frame: np.ndarray,
        masks: List[np.ndarray],
    ) -> np.ndarray:
        if not masks:
            return frame

        combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        for mask in masks:
            if mask is None:
                continue

            if mask.dtype != np.uint8:
                mask = mask.astype(np.uint8)

            combined_mask = np.maximum(combined_mask, mask)

        if combined_mask.max() == 0:
            return frame

        blurred_full = cv2.GaussianBlur(frame, self.ksize, 0)

        mask_3 = combined_mask[:, :, None] > 0
        frame = np.where(mask_3, blurred_full, frame)

        return frame

    def process(
        self,
        frame: np.ndarray,
        masks: Optional[List[np.ndarray]] = None,
    ) -> np.ndarray:
        if masks:
            return self.blur_combined_mask(frame, masks)
        return frame