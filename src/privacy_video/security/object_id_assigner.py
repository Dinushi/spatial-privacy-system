from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class ActiveTrack:
    object_id: str
    label: str
    bbox: Tuple[int, int, int, int]
    last_frame_idx: int

# compute IoU between two boxes.
def iou_xyxy(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih

    a_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    b_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = a_area + b_area - inter

    return inter / union if union > 0 else 0.0


class StableObjectIdAssigner:
    def __init__(self, iou_threshold: float = 0.3, max_gap_frames: int = 15) -> None:
        self.iou_threshold = iou_threshold # minimum overlap of bbox to consider “same object”
        self.max_gap_frames = max_gap_frames # how long a track can disappear before we stop matching to it
        self.tracks: List[ActiveTrack] = [] # list of active remembered objects
        self.counter = 1

    def assign(
        self,
        frame_idx: int,
        label: str,
        bbox: Optional[Tuple[int, int, int, int]],
    ) -> str:
        # If there is no box, it cannot compare spatially, so it just creates a new object ID.
        if bbox is None:
            obj_id = f"obj_{self.counter}"
            self.counter += 1
            return obj_id

        best_idx = None
        best_iou = -1.0

        for idx, tr in enumerate(self.tracks): # For each remembered track, check:
            if tr.label != label: # Same label?
                continue
            if frame_idx - tr.last_frame_idx > self.max_gap_frames: # Seen recently enough?
                continue
            score = iou_xyxy(bbox, tr.bbox) # If both checks pass, compute overlap score.
            if score > best_iou:
                best_iou = score
                best_idx = idx

        # If a good enough previous track is found:
        if best_idx is not None and best_iou >= self.iou_threshold:
            self.tracks[best_idx].bbox = bbox
            self.tracks[best_idx].last_frame_idx = frame_idx
            return self.tracks[best_idx].object_id
        
        # Otherwise create new object ID
        obj_id = f"obj_{self.counter}"
        self.counter += 1
        self.tracks.append(
            ActiveTrack(
                object_id=obj_id,
                label=label,
                bbox=bbox,
                last_frame_idx=frame_idx,
            )
        )
        return obj_id