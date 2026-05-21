from __future__ import annotations

from pathlib import Path

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import cv2
import time

from privacy_video.metadata.json_writer import JSONWriter
from privacy_video.models.SAM_result import FrameDetections
# from privacy_video.processing.blur_processor import BlurProcessor
# from privacy_video.processing.blur_processor_single import CombinedMaskBlurProcessor
# from privacy_video.processing.blur_processor_single_roi import CombinedMaskBBoxROIBlurProcessor
from privacy_video.processing.blur_processor_pixelate import CombinedMaskPixelateProcessor
from privacy_video.processing.crop_extractor import CropExtractor
from privacy_video.processing.privacy_prompt_processor import PrivacyPromptProcessor
from privacy_video.processing.sam_processor import SAMProcessor
from privacy_video.processing.fast_sam_processor import FastSAMProcessor
# from privacy_video.processing.fast_sam_trackingprocessor import FastSAMTrackProcessor
from privacy_video.utils.file_utils import is_image_file, is_video_file, get_video_specs

from common.security import (
    encrypt_json_hybrid,
    generate_aes256_key,
    load_public_key,
    rsa_wrap_key,
)
from privacy_video.security.media_embedder import MediaEmbedder
from privacy_video.security.object_id_assigner import StableObjectIdAssigner
from privacy_video.security.json_generator import generate_metadata_json, build_final_embedding_payload
from privacy_video.security.region_packager import (
    PrivateRegionEntryInput,
    build_private_region_entry
)

def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    JSONWriter(path).write(payload)


def _make_media_id() -> str:
    now = datetime.now(timezone.utc)
    return f"media_{now.strftime('%Y%m%d_%H%M%S')}"

def create_SAM_processor_object(SAM_type, video_stride, model_path):
    if (SAM_type == "SAM3"):
        return SAMProcessor(
            model_path=model_path,
            conf=0.25,
            imgsz=392, # imgsz=640, imgsz=384, # Note: must be multiple of max stride 14
            half=False,  # keep stable for now
            vid_stride = video_stride
        )
    else:
        return FastSAMProcessor(
            model_path=model_path,
            conf=0.25,
            imgsz=1024,
            half=True,
            # vvid_stride = video_stride
        )
        # sam_processor = FastSAMTrackProcessor(
        #     model_path=model_path,
        #     imgsz=640,
        #     conf=0.25,
        #     vid_stride=2,
        # )


def run_privacy_pipeline(
    source_path: str | Path,
    model_path: str | Path,
    output_root: str | Path,
    SAM_type: str = None,
    prompts: Optional[List[str]] = None,
    crop_mode: str = "mask",
    public_key_path: str | Path | None = None,
    video_stride: int = 1,
    save_payloads: bool = False,
) -> Dict[str, Any]:
    source_path = str(source_path)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if public_key_path is None:
        raise ValueError("public_key_path is required for encryption.")
    public_key = load_public_key(public_key_path)
    # a unique id for a processed media asset, which connect all generated files
    media_id = _make_media_id()
    crop_extractor = CropExtractor(output_root / "extracted_private_objects")
    crop_paths: List[Optional[str]] = []
    sam_total_time = 0.0
    post_processing_total_time = 0.0

    print("Create SAM Processor Object")
    sam_processor = create_SAM_processor_object(SAM_type, video_stride, model_path)
  
    # blur_processor = BlurProcessor()
    # blur_processor = CombinedMaskBlurProcessor()
    # blur_processor = CombinedMaskBBoxROIBlurProcessor()
    blur_processor = CombinedMaskPixelateProcessor(pixel_size=40)

    object_assigner = StableObjectIdAssigner()

    private_region_entries: List[Dict[str, Any]] = []
    region_counter = 1

    # object_keys: Dict[str, bytes] = {}
    # def get_or_create_object_key(object_id: str) -> bytes:
    #     if object_id not in object_keys:
    #         object_keys[object_id] = generate_aes256_key()
    #     return object_keys[object_id]
    
    seen_labels: set[str] = set()
    framesIDs_per_label: Dict[str, set[int]] = defaultdict(set) # hold the frame numbers at which each label is detected
        
    regionIDs_per_label: Dict[str, List[str]] = defaultdict(list)

    AES_keys_per_label: Dict[str, bytes] = {}
    def get_or_create_AES_key_for_label(label: str) -> bytes:
        if label not in AES_keys_per_label:
            AES_keys_per_label[label] = generate_aes256_key()
        return AES_keys_per_label[label]

    # Get the privacy prompts
    prompts = prompts or PrivacyPromptProcessor().process()

    if is_image_file(source_path):
        media_type = "image"
        print("Start processing Image ->")
        frame_det, sam_total_time = sam_processor.process_image(source_path, prompts, object_assigner)


        original_img = cv2.imread(source_path)
        if original_img is None:
            raise RuntimeError(f"Failed to read image: {source_path}")
        
        blurred_path = output_root / "blurred_output.png"

        post_start_time = time.time()
        frame_masks = []
        frame_bboxes = []
        for det in frame_det.objects:
    
            if det.mask is None or det.bbox is None:
                continue
            label = det.label
            bbox = tuple(det.bbox)
            print(f"\t\tApply privacy on object label: {label}")

            if label not in seen_labels:
                seen_labels.add(label)
                print(f"\t\tNew object label seen: {label}")
            frame_masks.append(det.mask)
            frame_bboxes.append(bbox)

            # if crop_mode == "mask" and det.mask is not None:
            #     crop = crop_extractor.extract_mask_crop(original_img, det.mask, bbox=det.bbox)
            # elif det.bbox is not None:
            #     crop = crop_extractor.extract_bbox_crop(original_img, det.bbox)
            # else:
            #     continue
            crop = crop_extractor.extract_mask_crop(original_img, det.mask, bbox=det.bbox)
            ##TODO: comment when run for evaluations
            # crop_extractor.save_crop(
            #     crop=crop,
            #     frame_idx=0,
            #     object_idx=det.object_idx,
            #     label=det.label,
            # )

            private_region_id = f"reg_{region_counter}"
            region_counter += 1

            encryption_key = get_or_create_AES_key_for_label(label)
            private_region_entry = build_private_region_entry(
                PrivateRegionEntryInput(
                    region_id=private_region_id,
                    object_id=label,
                    frame_idx=0,
                    bbox=bbox,
                    crop=crop,
                    placement_mode=crop_mode,
                    mask=det.mask,
                ),
                encryption_key,
            )
            private_region_entries.append(private_region_entry)
            # seperately collect private regions IDs label-wise
            regionIDs_per_label[label].append(private_region_id)

        protected_frame = blur_processor.process(original_img, masks=frame_masks, bboxes=frame_bboxes)
        post_processing_total_time = time.time() - post_start_time
        cv2.imwrite(str(blurred_path), protected_frame)

    elif is_video_file(source_path):
        # open the original source file
        media_type = "video"
        cap = cv2.VideoCapture(source_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open input video: {source_path}")
        fps, width, height, total_frames = get_video_specs(cap)
        
        # open a writer to save the privacy-preserved output
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
        try:
            for frame_det, sam_time in sam_processor.process_video_stream(source_path, prompts):
                sam_total_time += sam_time
                print(f"SAM processing time for frame {frame_det.frame_idx} : {sam_time:.2f}s")

                post_start_time = time.time()
                original_frame_idx = frame_det.frame_idx
                cap.set(cv2.CAP_PROP_POS_FRAMES, original_frame_idx) # TODO: could remove this make the processing futher fast
                print(f"Apply post processing steps based on SAM output on frame: {original_frame_idx}")

                for i in range(video_stride):
                    ok, original_img = cap.read()
                    if not ok or original_img is None:
                        break

                    current_frame_id = original_frame_idx + i
                    print(f"\tCurrent frame ID to propagate detections: {current_frame_id}")

                    protected_frame = original_img.copy()

                    # collect all these togther to later apply masking
                    frame_masks = []
                    frame_bboxes = []
                    for det in frame_det.objects:
                        if det.mask is None or det.bbox is None:
                            continue
                        label = det.label
                        bbox = tuple(det.bbox)
                        print(f"\t\tApply privacy on object label: {label}")

                        if label not in seen_labels:
                            seen_labels.add(label)
                            print(f"\t\tNew object label seen: {label}")
                        framesIDs_per_label[label].add(current_frame_id)

                        frame_masks.append(det.mask)
                        frame_bboxes.append(bbox)

                        crop = crop_extractor.extract_mask_crop(original_img, det.mask, bbox=det.bbox)
                        #this is an addition step to save cropped version of every masked object for verification purposes
                        #TODO: comment when run for evaluations
                        # crop_path = crop_extractor.save_crop(
                        #     crop=crop,
                        #     frame_idx=current_frame_id,
                        #     object_idx=det.object_idx,
                        #     label=det.label,
                        # )

                        # Maintain a seperate Id for each crop, Maintaining <Frame_Id><Lable><InstanceOccurance> is hard, so, adopt simply a unique id for each crop
                        private_region_id = f"reg_{region_counter}"
                        region_counter += 1

                        encryption_key = get_or_create_AES_key_for_label(label)
                        private_region_entry = build_private_region_entry(
                            PrivateRegionEntryInput(
                                region_id=private_region_id,
                                object_id=label,          # label used as encryption identity
                                frame_idx=current_frame_id,
                                bbox=bbox,
                                crop=crop,
                                placement_mode=crop_mode,
                                mask=det.mask,
                            ), encryption_key)
                        private_region_entries.append(private_region_entry)
                        # seperately collect private regions IDs label-wise
                        regionIDs_per_label[label].append(private_region_id)

                    # Apply pixelate blur at once for all the object masks
                    protected_frame = blur_processor.process(
                        protected_frame,
                        masks=frame_masks,
                        bboxes=frame_bboxes,
                    )
                    print(f"\tApplied privacy blurring on original frame")
                    writer.write(protected_frame)
                post_processing_time = time.time() - post_start_time
                post_processing_total_time += post_processing_time

        finally:
            cap.release()
            writer.release()   

    print(f"SAM Time (seconds) : {(sam_total_time):.2f}")
    print(f"Post-Processing Time (seconds): {post_processing_total_time:.2f}")

    # ---------------------------------------------------------
    # 1. save metadata file
    # ---------------------------------------------------------

    metadata = generate_metadata_json(
            media_id=media_id,
            media_type=media_type,
            source_path=source_path,
            blurred_output_path=str(blurred_path),
            fps=fps if media_type == "video" else None,
            width=width if media_type == "video" else None,
            height=height if media_type == "video" else None,
            total_frames=total_frames if media_type == "video" else None,
            video_stride=video_stride if media_type == "video" else None,
            seen_labels=seen_labels,
            framesIDs_per_label=framesIDs_per_label,
            regionIDs_per_label=regionIDs_per_label,
            sam_total_time=sam_total_time,
            post_processing_total_time=post_processing_total_time,
    )
    if save_payloads:
         _save_json(output_root / "metadata.json", metadata)

    # ---------------------------------------------------------
    # 2. save encrypted private regions file
    #    This contains ciphertext crops, bbox, masks, frame_idx.
    #    Organized under labels
    # ---------------------------------------------------------

    encrypted_private_regions = {
            "version": 1,
            "media_id": media_id,
            "media_type": media_type,
            "placement_mode": crop_mode,
            "encrypted_private_regions": private_region_entries,
    }
    if save_payloads:
            encrypted_regions_path = output_root / "encrypted_private_regions.json"
            _save_json(encrypted_regions_path, encrypted_private_regions)

    # ---------------------------------------------------------
    # 3. Key registry plaintext
    #    Then encrypt whole registry using device/server public key.
    # ---------------------------------------------------------

    AESkey_registry_plain = {
        "version": 1,
        "media_id": media_id,
        "key_scope": "label",
        "wrapped_label_keys": [
            {
                "label": label,
                "key_id": f"key_{label}",
                "sym_alg": "AES-256-GCM",
                "raw_key_base64_note": "raw AES key is encrypted inside registry; not exposed directly",
                "allowed_region_ids": regionIDs_per_label[label],
                "wrapped_key": rsa_wrap_key(public_key, key),
                "wrap_alg": "RSA-OAEP-SHA256",
            }
            for label, key in AES_keys_per_label.items()
        ],
    }
    if save_payloads:
        # this saved simply for verification only
        key_registry_path = output_root / "AESkey_registry_plain.json"
        _save_json(key_registry_path, AESkey_registry_plain)

    # This is the content that will be embedded into the saved media, which has AES encryption wrap keys in encrypted format 
    # encrypted using device's public key and the AES keys can be retrived only after decryption by the device using its public key
    encrypted_AES_encryption_key_registry = {
        "version": 1,
        "media_id": media_id,
        "enc_alg": "hybrid-json",
        **encrypt_json_hybrid(public_key, AESkey_registry_plain),
    }
    if save_payloads:
        encrypte_key_registry_path = output_root / "AESkey_registry_encrypted.json"
        _save_json(encrypte_key_registry_path, encrypted_AES_encryption_key_registry)

    # -----------------------------------------------------------
    # 4. Embed all three files into the same blurred media output
    # -----------------------------------------------------------

    payload_bytes = build_final_embedding_payload(
        media_id=media_id,
        metadata=metadata,
        encrypted_private_regions=encrypted_private_regions,
        encrypted_key_registry=encrypted_AES_encryption_key_registry,
    )
    MediaEmbedder.embed_payload_in_file(
        media_path=blurred_path,
        payload_bytes=payload_bytes,
    )


     ####
        #Below contains the old version of the code
        # which collected all SAM masks for all the frames and later apply masking
        # my analysis showed that this might cause memory overflow when collecting larger masks
    ####

    '''
        print("Start processing Video using SAM ----------------------->")
        start_time = time.time()
        frame_detections = sam_processor.process_video(source_path, prompts, object_assigner, stream=True) # stream=False
        total_sec_for_sam_exec = time.time() - start_time

        # cap = cv2.VideoCapture(source_path)
        # if not cap.isOpened():
        #     raise RuntimeError(f"Failed to open input video: {source_path}")

        # fps = float(cap.get(cv2.CAP_PROP_FPS))
        # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # if fps <= 0:
        #     fps = 30.0

        # blurred_path = output_root / "blurred_output.mp4"
        # writer = cv2.VideoWriter(
        #     str(blurred_path),
        #     cv2.VideoWriter_fourcc(*"mp4v"),
        #     fps,
        #     (width, height),
        # )

        # if not writer.isOpened():
        #     cap.release()
        #     raise RuntimeError(f"Failed to open output video writer: {blurred_path}")

        frames_meta: List[Dict[str, Any]] = []

        print("Apply SAM detections as actual masks on video ------------------------>")
        start_time_masking = time.time()
        try:
            for frame_det in frame_detections:
                original_frame_idx = frame_det.frame_idx
                print(f"Frame index of the current frame detection from SAM: {original_frame_idx}")
                cap.set(cv2.CAP_PROP_POS_FRAMES, original_frame_idx) # ask reader to jump to a specific frame number

                # this ensures the original frames are processed one by one and propogate detetections if frame-skip has enabled in SAM detections
                for i in range(video_stride):
                    ok, original = cap.read() # reads the next frame
                    if not ok or original is None:
                        break

                    current_frame_id = original_frame_idx + i
                    print(f"\tCurrent frame ID to propagate data: {current_frame_id}")
                    current_frame = original.copy()
                    crop_paths: List[Optional[str]] = []

                    protected_frame = current_frame
                    # collect the masks and bounding boxes for the objects detected in the frame togther and later apply masking at once
                    frame_masks = []
                    frame_bboxes = []
                    for det in frame_det.objects:

                        object_id = det.custom_tracked_object_id
                        bbox = tuple(det.bbox) if det.bbox is not None else None
                        if crop_mode == "mask" and det.mask is not None:
                            # start_time = time.time()
                            # protected_frame = blur_processor.process(protected_frame, mask=det.mask)
                            # print(f"Masking time: {(time.time() - start_time)/60:.2f} minutes")
                            frame_masks.append(det.mask)
                            start_time = time.time()
                            crop = crop_extractor.extract_mask_crop(original, det.mask, bbox=det.bbox)
                            print(f"Crop extraction time: {(time.time() - start_time):.6f} s")

                            frame_bboxes.append(tuple(det.bbox))
                        elif det.bbox is not None:
                            # protected_frame = blur_processor.process(protected_frame, bbox=det.bbox)
                            crop = crop_extractor.extract_bbox_crop(original, det.bbox)
                        else:
                            crop_paths.append(None)
                            continue

                        #this is an addition step to save cropped version of every masked object for verification purposes
                        # TODO: comment when run for evaluations
                        # crop_path = crop_extractor.save_crop(
                        #     crop=crop,
                        #     frame_idx=current_frame_id,
                        #     object_idx=det.object_idx,
                        #     label=det.label,
                        # )
                        # crop_paths.append(crop_path)

                        # create a entry for each region
                        region_id = f"reg_{region_counter}"
                        region_counter += 1
                        obj_key = get_or_create_object_key(object_id)
                        region_entry = build_private_region_entry(
                            PrivateRegionEntryInput(
                                region_id=region_id,
                                object_id=object_id,
                                frame_idx=frame_det.frame_idx,
                                bbox=bbox,
                                crop=crop,
                                placement_mode=crop_mode,
                                mask=det.mask,
                            ),
                            obj_key,
                        )
                        private_region_entries.append(region_entry)
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

                    start_time = time.time()
                    # protected_frame = blur_processor.process(protected_frame, masks=frame_masks)
                    # print(f"Masking time (all object masked collected together and masking applied once): {(time.time() - start_time)/60:.2f} minutes")
                    protected_frame = blur_processor.process(protected_frame, masks=frame_masks,bboxes=frame_bboxes)
                    print(f"Masking time (collected masks on union RoI) seconds: {(time.time() - start_time):.2f}")
                    writer.write(protected_frame)

                frame_meta = _frame_detection_to_metadata(frame_det, crop_paths)
                frame_meta["timestamp_sec"] = frame_det.frame_idx / fps
                frames_meta.append(frame_meta)
        finally:
            cap.release()
            writer.release()
    
        print(f"SAM Time (seconds) : {(total_sec_for_sam_exec):.2f}")
        print(f"Post-Processing Time (seconds): {(time.time()- start_time_masking):.2f}")
        frames_meta_for_verification = frames_meta
        media_type = "video"
        
        # metadata = {
        #     "input_type": "video",
        #     "input_path": source_path,
        #     "prompts": prompts,
        #     "blurred_output_path": str(blurred_path),
        #     "fps": fps,
        #     "width": width,
        #     "height": height,
        #     "frames": frames_meta,
        # }

        # JSONWriter(output_root / "metadata.json").write(metadata)
        # return metadata
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
        encrypted_regions=private_region_entries,
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
    '''

