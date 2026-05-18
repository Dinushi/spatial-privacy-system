from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from .blur_processor import BlurProcessor

class CombinedMaskBBoxROIBlurProcessor(BlurProcessor):
    """
    Combines masks, but uses available bboxes to find ROI (Region of Interest) quickly.
    Blurs only the union bbox area, then applies combined mask inside it.
    """

    def blur_combined_mask_bbox_roi(
        self,
        frame: np.ndarray,
        masks: List[np.ndarray],
        bboxes: List[Tuple[int, int, int, int]],
    ) -> np.ndarray:
        if not masks or not bboxes:
            return frame

        h, w = frame.shape[:2]

        x1 = min(b[0] for b in bboxes)
        y1 = min(b[1] for b in bboxes)
        x2 = max(b[2] for b in bboxes)
        y2 = max(b[3] for b in bboxes)

        x1 = max(0, min(int(x1), w - 1))
        x2 = max(0, min(int(x2), w))
        y1 = max(0, min(int(y1), h - 1))
        y2 = max(0, min(int(y2), h))

        if x2 <= x1 or y2 <= y1:
            return frame

        # reates a view/reference into the original frame memory
        roi = frame[y1:y2, x1:x2]

        combined_roi_mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)

        for mask in masks:
            if mask is None:
                continue

            if mask.dtype != np.uint8:
                mask = mask.astype(np.uint8)

            local_mask = mask[y1:y2, x1:x2]
            combined_roi_mask = np.maximum(combined_roi_mask, local_mask)

        if combined_roi_mask.max() == 0:
            return frame

        # apply Gaussian blur only to a region of interest of the frame , which objects occur
        blurred_roi = cv2.GaussianBlur(roi, self.ksize, 0)

        mask_3 = combined_roi_mask[:, :, None] > 0
        # directly modifying the corresponding region inside frame
        roi[:] = np.where(mask_3, blurred_roi, roi)

        return frame

    def process(
        self,
        frame: np.ndarray,
        masks: List[np.ndarray],
        bboxes: List[Tuple[int, int, int, int]],
    ) -> np.ndarray:
        return self.blur_combined_mask_bbox_roi(frame, masks, bboxes)