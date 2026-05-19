from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from .blur_processor import BlurProcessor


class CombinedMaskPixelateProcessor(BlurProcessor):
    """
    Pixelates all masked objects together.
    Uses bboxes to compute a union ROI, then applies combined mask inside that ROI.
    """

    def __init__(
        self,
        pixel_size: int = 20,
        ksize: Tuple[int, int] = (101, 101),  # unused, kept for parent compatibility
    ) -> None:
        super().__init__(ksize=ksize)
        self.pixel_size = pixel_size

    def pixelate_combined_mask_bbox_roi(
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

        roi = frame[y1:y2, x1:x2]
        roi_h, roi_w = roi.shape[:2]

        combined_roi_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
        for mask in masks:
            if mask is None:
                continue

            if mask.dtype != np.uint8:
                mask = mask.astype(np.uint8)

            local_mask = mask[y1:y2, x1:x2]
            # merges all object masks together
            combined_roi_mask = np.maximum(combined_roi_mask, local_mask)

        if combined_roi_mask.max() == 0:
            return frame

        # Compute tiny resolution
        # TODO: change pixel size to contol the blur vs speed (tiny -> finer blocks, more details preserved ), (large -> giant blocks, fast, stronger privacy)
        small_w = max(1, roi_w // self.pixel_size)
        small_h = max(1, roi_h // self.pixel_size)

        # shrink/ compresses the ROI region to above size
        small = cv2.resize(
            roi,
            (small_w, small_h),
            interpolation=cv2.INTER_LINEAR,
        )
        # expands tiny image back to original size
        pixelated_roi = cv2.resize(
            small,
            (roi_w, roi_h),
            interpolation=cv2.INTER_NEAREST, # copy nearest pixel exactly (tiny pixel becomes a big square block -> creates the pixelated effect)
        )

        mask_3 = combined_roi_mask[:, :, None] > 0
        roi[:] = np.where(mask_3, pixelated_roi, roi)

        return frame

    def process(
        self,
        frame: np.ndarray,
        masks: List[np.ndarray],
        bboxes: List[Tuple[int, int, int, int]],
    ) -> np.ndarray:
        return self.pixelate_combined_mask_bbox_roi(
            frame=frame,
            masks=masks,
            bboxes=bboxes,
        )