from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

from privacy_video.metadata.json_writer import JSONWriter
from privacy_video.models.SAM_result import FrameDetections
from privacy_video.processing.blur_processor import BlurProcessor
from privacy_video.processing.crop_extractor import CropExtractor
from privacy_video.processing.privacy_prompt_processor import PrivacyPromptProcessor
from privacy_video.processing.sam_processor import SAMProcessor
from privacy_video.utils.file_utils import is_image_file, is_video_file


def _frame_detection_to_metadata(frame_det: FrameDetections, crop_paths: List[Optional[str]]) -> Dict[str, Any]:
    objects_meta: List[Dict[str, Any]] = []

    for det, crop_path in zip(frame_det.objects, crop_paths):
        objects_meta.append(
            {
                "object_idx": det.object_idx,
                "label": det.label,
                "class_id": det.class_id,
                "confidence": det.confidence,
                "bbox": list(det.bbox) if det.bbox is not None else None,
                "has_mask": det.mask is not None,
                "extracted_crop_path": crop_path,
            }
        )

    return {
        "frame_idx": frame_det.frame_idx,
        "source_path": frame_det.source_path,
        "orig_shape": list(frame_det.orig_shape),
        "objects": objects_meta,
    }


def run_privacy_pipeline(
    source_path: str | Path,
    model_path: str | Path,
    output_root: str | Path,
    prompts: Optional[List[str]] = None,
) -> Dict[str, Any]:
    source_path = str(source_path)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    prompts = prompts or PrivacyPromptProcessor().process()

    sam_processor = SAMProcessor(
        model_path=model_path,
        conf=0.25,
        imgsz=640,
        half=False,  # keep stable for now
    )
    blur_processor = BlurProcessor()
    crop_extractor = CropExtractor(output_root / "extracted_private_objects")

    if is_image_file(source_path):
        frame_det = sam_processor.process_image(source_path, prompts)

        original = cv2.imread(source_path)
        if original is None:
            raise RuntimeError(f"Failed to read image: {source_path}")

        frame = original.copy()
        crop_paths: List[Optional[str]] = []

        for det in frame_det.objects:
            if det.mask is not None:
                frame = blur_processor.process(frame, mask=det.mask)
                crop = crop_extractor.extract_mask_crop(original, det.mask, bbox=det.bbox)
            elif det.bbox is not None:
                frame = blur_processor.process(frame, bbox=det.bbox)
                crop = crop_extractor.extract_bbox_crop(original, det.bbox)
            else:
                crop_paths.append(None)
                continue

            crop_path = crop_extractor.save_crop(
                crop=crop,
                frame_idx=0,
                object_idx=det.object_idx,
                label=det.label,
            )
            crop_paths.append(crop_path)

        blurred_path = output_root / "blurred_output.png"
        cv2.imwrite(str(blurred_path), frame)

        metadata = {
            "input_type": "image",
            "input_path": source_path,
            "prompts": prompts,
            "blurred_output_path": str(blurred_path),
            "frame": _frame_detection_to_metadata(frame_det, crop_paths),
        }

        JSONWriter(output_root / "metadata.json").write(metadata)
        return metadata

    if is_video_file(source_path):
        frame_detections = sam_processor.process_video(source_path, prompts, stream=False)

        cap = cv2.VideoCapture(source_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open input video: {source_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if fps <= 0:
            fps = 30.0

        blurred_path = output_root / "blurred_output.mp4"
        writer = cv2.VideoWriter(
            str(blurred_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Failed to open output video writer: {blurred_path}")

        frames_meta: List[Dict[str, Any]] = []

        try:
            for frame_det in frame_detections:
                ok, original = cap.read()
                if not ok or original is None:
                    break

                frame = original.copy()
                crop_paths: List[Optional[str]] = []

                for det in frame_det.objects:
                    if det.mask is not None:
                        frame = blur_processor.process(frame, mask=det.mask)
                        crop = crop_extractor.extract_mask_crop(original, det.mask, bbox=det.bbox)
                    elif det.bbox is not None:
                        frame = blur_processor.process(frame, bbox=det.bbox)
                        crop = crop_extractor.extract_bbox_crop(original, det.bbox)
                    else:
                        crop_paths.append(None)
                        continue

                    crop_path = crop_extractor.save_crop(
                        crop=crop,
                        frame_idx=frame_det.frame_idx,
                        object_idx=det.object_idx,
                        label=det.label,
                    )
                    crop_paths.append(crop_path)

                writer.write(frame)

                frame_meta = _frame_detection_to_metadata(frame_det, crop_paths)
                frame_meta["timestamp_sec"] = frame_det.frame_idx / fps
                frames_meta.append(frame_meta)
        finally:
            cap.release()
            writer.release()

        metadata = {
            "input_type": "video",
            "input_path": source_path,
            "prompts": prompts,
            "blurred_output_path": str(blurred_path),
            "fps": fps,
            "width": width,
            "height": height,
            "frames": frames_meta,
        }

        JSONWriter(output_root / "metadata.json").write(metadata)
        return metadata

    raise ValueError(f"Unsupported input type: {source_path}")