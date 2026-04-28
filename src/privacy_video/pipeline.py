from __future__ import annotations

from pathlib import Path

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from privacy_video.metadata.json_writer import JSONWriter
from privacy_video.models.SAM_result import FrameDetections
from privacy_video.processing.blur_processor import BlurProcessor
from privacy_video.processing.crop_extractor import CropExtractor
from privacy_video.processing.privacy_prompt_processor import PrivacyPromptProcessor
from privacy_video.processing.sam_processor import SAMProcessor
from privacy_video.utils.file_utils import is_image_file, is_video_file

from common.security import (
    encrypt_json_hybrid,
    generate_aes256_key,
    load_public_key,
    rsa_wrap_key,
)
from privacy_video.security.media_embedder import MediaEmbedder
from privacy_video.security.object_id_assigner import StableObjectIdAssigner
from privacy_video.security.region_packager import (
    RegionEntryInput,
    build_region_entry,
    build_region_package,
)

def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    JSONWriter(path).write(payload)


def _make_media_id() -> str:
    now = datetime.now(timezone.utc)
    return f"media_{now.strftime('%Y%m%d_%H%M%S')}"


def _frame_detection_to_metadata(frame_det: FrameDetections, crop_paths: List[Optional[str]], frame_idx=0) -> Dict[str, Any]:
    objects_meta: List[Dict[str, Any]] = []

    for det, crop_path in zip(frame_det.objects, crop_paths):
        objects_meta.append(
            {
                "object_idx": det.object_idx,
                "custom_tracked_object_id": det.custom_tracked_object_id,
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
    crop_mode: str = "mask",
    public_key_path: str | Path | None = None,
    embed_payloads: bool = False,
) -> Dict[str, Any]:
    source_path = str(source_path)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if public_key_path is None:
        raise ValueError("public_key_path is required for encryption.")
    public_key = load_public_key(public_key_path)
    # a unique id for a processed media asset, which connect all generated files
    media_id = _make_media_id()

    # create required process objects
    sam_processor = SAMProcessor(
        model_path=model_path,
        conf=0.25,
        imgsz=640,
        half=False,  # keep stable for now
    )
    blur_processor = BlurProcessor()
    crop_extractor = CropExtractor(output_root / "extracted_private_objects")
    object_assigner = StableObjectIdAssigner()

    region_entries: List[Dict[str, Any]] = []
    object_keys: Dict[str, bytes] = {}
    object_regions: Dict[str, List[str]] = defaultdict(list)
    object_meta: Dict[str, Dict[str, Any]] = {}
    region_counter = 1

    def get_or_create_object_key(object_id: str) -> bytes:
        if object_id not in object_keys:
            object_keys[object_id] = generate_aes256_key()
        return object_keys[object_id]

    # 1. get the privacy prompts
    prompts = prompts or PrivacyPromptProcessor().process()

    if is_image_file(source_path):
        print("Start processing Image ->")
        frame_det = sam_processor.process_image(source_path, prompts, object_assigner)

        original = cv2.imread(source_path)
        if original is None:
            raise RuntimeError(f"Failed to read image: {source_path}")

        frame = original.copy()
        crop_paths: List[Optional[str]] = []

        for det in frame_det.objects:
            bbox = tuple(det.bbox) if det.bbox is not None else None
            # object_id = object_assigner.assign(
            #     frame_idx=0, # always 0 for a image
            #     label=det.label,
            #     bbox=bbox,
            # )
            object_id = det.custom_tracked_object_id
        
            if crop_mode == "mask" and det.mask is not None:
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

            region_id = f"reg_{region_counter}"
            region_counter += 1

            obj_key = get_or_create_object_key(object_id)
            region_entry = build_region_entry(
                RegionEntryInput(
                    region_id=region_id,
                    object_id=object_id,
                    frame_idx=0,
                    bbox=bbox,
                    crop=crop,
                    placement_mode=crop_mode,
                    mask=det.mask,
                ),
                obj_key,
            )
            region_entries.append(region_entry)
            object_regions[object_id].append(region_id)

            if object_id not in object_meta:
                object_meta[object_id] = {
                    "object_id": object_id,
                    "label": det.label,
                    "class_id": det.class_id,
                    "first_frame_idx": 0,
                    "last_frame_idx": 0,
                    "region_ids": [],
                }

        for obj_id in object_meta:
            object_meta[obj_id]["region_ids"] = object_regions[obj_id]   

        blurred_path = output_root / "blurred_output.png"
        cv2.imwrite(str(blurred_path), frame)

        media_type = "image"

        # here building the older version of metadata file for verification purposes ONLY
        # metadata = {
        #     "input_type": "image",
        #     "input_path": source_path,
        #     "prompts": prompts,
        #     "blurred_output_path": str(blurred_path),
        #     "frame": _frame_detection_to_metadata(frame_det, crop_paths),
        # }
        frames_meta_for_verification = _frame_detection_to_metadata(frame_det, crop_paths),

        # JSONWriter(output_root / "metadata.json").write(metadata)
        # return metadata
      
    elif is_video_file(source_path):
        print("Start processing Video ->")
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
                    if crop_mode == "mask" and det.mask is None:
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
    else:
        raise ValueError(f"Unsupported input type: {source_path}")
    
    # 1. Encrypted metadata file, save a cypher text of metadata
    metadata_plain = {
        "version": 1,
        "media_id": media_id,
        "media_type": media_type,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objects": list(object_meta.values()),
    }
    encrypted_metadata_file = {
        "version": 1,
        "media_id": media_id,
        **encrypt_json_hybrid(public_key, metadata_plain),
    }
    _save_json(output_root / "encrypted_metadata_file.json", encrypted_metadata_file)

    # 2. Embedded encrypted region package - this holds the encrypted cipher text of private objects
    region_package = build_region_package(
        media_id=media_id,
        media_type=media_type,
        placement_mode=crop_mode,
        encrypted_regions=region_entries,
    )
    _save_json(output_root / "embedded_encrypted_region_package.json", region_package)

    # 3. Object symmetric key registry
    wrapped_registry = {
        "version": 1,
        "media_id": media_id,
        "wrapped_object_keys": [
            {
                "object_id": object_id,
                "wrap_alg": "RSA-OAEP-SHA256",
                "wrapped_key": rsa_wrap_key(public_key, key),
                "allowed_region_ids": object_regions[object_id],
            }
            for object_id, key in object_keys.items()
        ],
    }
    _save_json(output_root / "object_symmetric_key_registry.json", wrapped_registry)

    if embed_payloads:
        payload_bytes = MediaEmbedder.build_payload(
            encrypted_metadata_file=encrypted_metadata_file,
            region_package=region_package,
            object_key_registry=wrapped_registry,
        )
        MediaEmbedder.embed_payload_in_file(blurred_path, payload_bytes)

    metadata = {
        "input_type": media_type,
        "input_path": source_path,
        "prompts": prompts,
        "media_id": media_id,
        "blurred_output_path": str(blurred_path),
        "encrypted_metadata_file_path": str(output_root / "encrypted_metadata_file.json"),
        "embedded_encrypted_region_package_path": str(output_root / "embedded_encrypted_region_package.json"),
        "object_symmetric_key_registry_path": str(output_root / "object_symmetric_key_registry.json"),
        "embedded_in_blurred_output": embed_payloads,
        "frames": frames_meta_for_verification,
    }
    _save_json(output_root / "general_metadata.json", metadata)
    return metadata


