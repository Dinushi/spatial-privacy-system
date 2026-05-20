from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
import time


from privacy_video.models.SAM_result import DetectedObject, FrameDetections
from privacy_video.security.object_id_assigner import StableObjectIdAssigner

from ultralytics.models.sam import SAM3SemanticPredictor
from typing import Iterator


class SAMProcessor:
    """
    Only runs SAM and converts Ultralytics Results into normalized objects.
    No blur, no metadata writing, no crop extraction here.
    """

    def __init__(
        self,
        model_path: str | Path,
        conf: float = 0.25,
        imgsz: int = 640,
        half: bool = False,
        vid_stride: int = 1,
    ) -> None:
        self.model_path = str(model_path)
        self.conf = conf
        self.imgsz = imgsz
        self.half = half
        self.vid_stride = vid_stride

    def _make_image_predictor(self):
   
        overrides = dict(
            conf=self.conf,
            task="segment",
            mode="predict",
            model=self.model_path,
            half=self.half,
            save=True,
        )
        return SAM3SemanticPredictor(overrides=overrides)

    def _make_video_predictor(self):
        from ultralytics.models.sam import SAM3VideoSemanticPredictor

        overrides = dict(
            conf=self.conf,
            task="segment",
            mode="predict",
            imgsz=self.imgsz,
            model=self.model_path,
            half=self.half,
            vid_stride=self.vid_stride, # to introduce any frame skip
            save=False,
        )
        return SAM3VideoSemanticPredictor(overrides=overrides)

    def _extract_label(self, result: Any, cls_id: int, det_idx: int) -> str:
        names = getattr(result, "names", None)

        if isinstance(names, dict):
            return str(names.get(cls_id, f"object_{det_idx}"))

        if isinstance(names, list) and 0 <= cls_id < len(names):
            return str(names[cls_id])
        # if no names exist, fallback
        return f"object_{det_idx}"

    def _resize_mask_to_orig(
        self,
        mask: np.ndarray,
        orig_shape: Tuple[int, int],
    ) -> np.ndarray:
        """
        mask from SAM video output may be smaller than original image/frame shape.
        Resize it back to original H, W.
        """
        import cv2

        orig_h, orig_w = orig_shape
        if mask.shape[0] == orig_h and mask.shape[1] == orig_w:
            return mask.astype(np.uint8)

        resized = cv2.resize(
            mask.astype(np.uint8),
            (orig_w, orig_h),
            interpolation=cv2.INTER_NEAREST,
        )
        return resized.astype(np.uint8)

    def _parse_result(
        self,
        result: Any,
        objectID_assigner : StableObjectIdAssigner,
        frame_idx: int,
        source_path: str,
    ) -> FrameDetections:
        boxes = getattr(result, "boxes", None)
        masks = getattr(result, "masks", None)
        orig_shape = tuple(getattr(result, "orig_shape", (0, 0)))

        xyxy_list = None
        conf_list = None
        cls_list = None
        mask_list = None

        if boxes is not None and getattr(boxes, "xyxy", None) is not None:
            xyxy_list = boxes.xyxy.detach().cpu().numpy().astype(int)

        if boxes is not None and getattr(boxes, "conf", None) is not None:
            conf_list = boxes.conf.detach().cpu().numpy()

        if boxes is not None and getattr(boxes, "cls", None) is not None:
            cls_list = boxes.cls.detach().cpu().numpy().astype(int)

        if masks is not None and getattr(masks, "data", None) is not None:
            mask_list = masks.data.detach().cpu().numpy()

        num_objects = 0
        if xyxy_list is not None:
            num_objects = len(xyxy_list)
        elif mask_list is not None:
            num_objects = len(mask_list)

        objects: List[DetectedObject] = []

        for obj_idx in range(num_objects):
            bbox: Optional[Tuple[int, int, int, int]] = None
            confidence: Optional[float] = None
            class_id: int = -1
            label: str = f"object_{obj_idx}"
            mask: Optional[np.ndarray] = None

            if xyxy_list is not None:
                x1, y1, x2, y2 = xyxy_list[obj_idx].tolist()
                bbox = (int(x1), int(y1), int(x2), int(y2))

            if conf_list is not None:
                confidence = float(conf_list[obj_idx])

            if cls_list is not None:
                class_id = int(cls_list[obj_idx])
                label = self._extract_label(result, class_id, obj_idx)

            if mask_list is not None:
                raw_mask = (mask_list[obj_idx] > 0.5).astype(np.uint8)
                mask = self._resize_mask_to_orig(raw_mask, orig_shape)

            # a custom object ID tracker number unique to each object accross each frame, this is a modication in introduced later
            # TODO: this might still not getting tracked, may be better to simply use the label to apply same key to single label
            object_id = objectID_assigner.assign(
                frame_idx=frame_idx,
                label=label,
                bbox=bbox,
            )
            # TODO: simplify the objectIdx assigment to a simple one
            objects.append(
                DetectedObject(
                    object_idx=obj_idx,
                    custom_tracked_object_id = object_id,
                    label=label,
                    class_id=class_id,
                    confidence=confidence,
                    bbox=bbox,
                    mask=mask,
                )
            )

        return FrameDetections(
            frame_idx=frame_idx,
            source_path=source_path,
            orig_shape=orig_shape,
            objects=objects,
        )

    def process_image(
        self,
        image_path: str | Path,
        prompts: List[str],
        objectID_assigner : StableObjectIdAssigner,
    ) -> FrameDetections:
        image_path = str(image_path)

        predictor = self._make_image_predictor()
        predictor.set_image(image_path)
        results = predictor(text=prompts)

        if not results:
            return FrameDetections(
                frame_idx=0,
                source_path=image_path,
                orig_shape=(0, 0),
                objects=[],
            )

        return self._parse_result(results[0], objectID_assigner, frame_idx=0, source_path=image_path)

    def process_video(
        self,
        video_path: str | Path,
        prompts: List[str],
        objectID_assigner : StableObjectIdAssigner,
        stream: bool = False,
    ) -> List[FrameDetections]:
        video_path = str(video_path)

        print("Some environment checks before video predictor creation =>")
        print("\tCUDA available:", torch.cuda.is_available())
        print("\tGPU count:", torch.cuda.device_count())
        if torch.cuda.is_available():
            print("\t\tCurrent GPU:", torch.cuda.current_device())
            print("\t\tGPU name:", torch.cuda.get_device_name(torch.cuda.current_device()))

        predictor = self._make_video_predictor()
        results = predictor(source=video_path, text=prompts, stream=stream)
        # When stream = TRUE -> result does NOT contain actual results yet, only a generator, Inference only happens when you ask for the next item in the loop below

        frame_results: List[FrameDetections] = []

        for sampled_idx, result in enumerate(results):
            # track back to original frame_idx set when frame skipping is applied, this multiplication reset the frameId by SAM to its original frame id in the full video
            original_frame_idx = sampled_idx * self.vid_stride

            frame_results.append(
                self._parse_result(result, objectID_assigner, frame_idx=original_frame_idx, source_path=video_path)
            )

        return frame_results
    
    def process_video_stream(
        self,
        video_path: str | Path,
        prompts: List[str],
    ) -> Iterator[FrameDetections]:
        video_path = str(video_path)

        predictor = self._make_video_predictor()
        results = predictor(source=video_path, text=prompts, stream=True)

        # for sampled_idx, result in enumerate(results): # IMPORTANT: SAM work happens when Python asks the generator for the next result, this is how we make use of steam
        #     original_frame_idx = sampled_idx * self.vid_stride

        #     yield self._parse_result_simple(
        #         result=result,
        #         frame_idx=original_frame_idx,
        #         source_path=video_path,
        #     )

        results_iter = iter(results)
        sampled_idx = 0

        while True:
            try:
                start = time.time()
                result = next(results_iter)   # IMPORTANT: SAM inference happens here
                sam_time = time.time() - start
            except StopIteration:
                break

            original_frame_idx = sampled_idx * self.vid_stride

            frame_det = self._parse_result_simple(
                result=result,
                frame_idx=original_frame_idx,
                source_path=video_path,
            )

            yield frame_det, sam_time

            sampled_idx += 1

    def _parse_result_simple(
        self,
        result: Any,
        frame_idx: int,
        source_path: str,
    ) -> FrameDetections:
        
        print(f"Parsing Frame ID : {frame_idx} ------>")
        boxes = getattr(result, "boxes", None)
        masks = getattr(result, "masks", None)
        orig_shape = tuple(getattr(result, "orig_shape", (0, 0)))

        xyxy_list = boxes.xyxy.detach().cpu().numpy().astype(int) if boxes is not None and boxes.xyxy is not None else None
        conf_list = boxes.conf.detach().cpu().numpy() if boxes is not None and boxes.conf is not None else None
        cls_list = boxes.cls.detach().cpu().numpy().astype(int) if boxes is not None and boxes.cls is not None else None
        mask_list = masks.data.detach().cpu().numpy() if masks is not None and masks.data is not None else None
  
        num_objects = len(xyxy_list) if xyxy_list is not None else len(mask_list) if mask_list is not None else 0
        print(f"\tNum of objects detected: {num_objects}")

        objects = []
        for obj_idx in range(num_objects):

            bbox = None
            confidence = None
            class_id = -1
            label = f"object_{obj_idx}"
            mask = None

            if xyxy_list is not None:
                x1, y1, x2, y2 = xyxy_list[obj_idx].tolist()
                bbox = (int(x1), int(y1), int(x2), int(y2))

            if conf_list is not None:
                confidence = float(conf_list[obj_idx])
            if cls_list is not None:
                class_id = int(cls_list[obj_idx])
                label = self._extract_label(result, class_id, obj_idx)
            # for now, the goald is simpler, metadata and reveal will be on label-level, no instance wise tracking is done within the lable
            # SAM does not provide direct instaance level tracking unless you provide more finer grained prompts for each
            print(f"\t\tExtracting detection info for object: class_id - {class_id}, label - {label}")

            if mask_list is not None:
                raw_mask = (mask_list[obj_idx] > 0.5).astype(np.uint8)
                mask = self._resize_mask_to_orig(raw_mask, orig_shape)

            objects.append(
                DetectedObject(
                    object_idx=obj_idx,
                    custom_tracked_object_id=label,  # label-level identity, TODO: need a approch to find intance level ids in future
                    label=label,
                    class_id=class_id,
                    confidence=confidence,
                    bbox=bbox,
                    mask=mask,
                )
            )
        # return the dection data of the frame immediately for the processing
        return FrameDetections(
            frame_idx=frame_idx,
            source_path=source_path,
            orig_shape=orig_shape,
            objects=objects,
        )