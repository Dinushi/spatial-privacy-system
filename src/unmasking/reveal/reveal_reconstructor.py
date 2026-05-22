from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import cv2

from unmasking.utils.crop_pasting_utils import paste_bbox_crop, paste_mask_crop


def group_decrypted_regions_by_frame(
    decrypted_regions: List[tuple[Dict[str, Any], Any]],
) -> Dict[int, List[tuple[Dict[str, Any], Any]]]:
    grouped = defaultdict(list)

    for region_entry, crop in decrypted_regions:
        frame_idx = int(region_entry["frame_idx"])
        grouped[frame_idx].append((region_entry, crop))

    return grouped


def reconstruct_image_with_revealed_regions(
    protected_image_path: str | Path,
    decrypted_regions: List[tuple[Dict[str, Any], Any]],
    output_path: str | Path,
    placement_mode: str,
) -> str:
    image = cv2.imread(str(protected_image_path))

    if image is None:
        raise RuntimeError(f"Failed to read protected image: {protected_image_path}")

    for region_entry, crop in decrypted_regions:
        if placement_mode == "mask":
            image = paste_mask_crop(image, crop, region_entry)
        else:
            image = paste_bbox_crop(image, crop, region_entry["bbox"])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_path), image)
    return str(output_path)


def reconstruct_video_with_revealed_regions(
    protected_video_path: str | Path,
    decrypted_regions: List[tuple[Dict[str, Any], Any]],
    output_path: str | Path,
    fps: float,
    width: int,
    height: int,
    placement_mode: str,
) -> str:
    grouped_regions = group_decrypted_regions_by_frame(decrypted_regions)

    cap = cv2.VideoCapture(str(protected_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open protected video: {protected_video_path}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open output video writer: {output_path}")

    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            if frame_idx in grouped_regions:
                for region_entry, crop in grouped_regions[frame_idx]:
                    if placement_mode == "mask":
                        frame = paste_mask_crop(frame, crop, region_entry)
                    else:
                        frame = paste_bbox_crop(frame, crop, region_entry["bbox"])

            writer.write(frame)
            frame_idx += 1

    finally:
        cap.release()
        writer.release()

    return str(output_path)