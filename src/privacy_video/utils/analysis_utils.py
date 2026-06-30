import cv2

BOX_COLOR = (0, 255, 0)
BOX_THICKNESS = 2

def draw_label(frame, text, x, y):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2

    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    y_text = max(y, th + 5)

    cv2.rectangle(
        frame,
        (x, y_text - th - baseline),
        (x + tw, y_text + baseline),
        BOX_COLOR,
        -1
    )
    cv2.putText(
        frame,
        text,
        (x, y_text),
        font,
        scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA
    )


def draw_detections_on_frame(frame, detections, bbox_format="xywh"):
    vis = frame.copy()
    h, w = vis.shape[:2]

    for det in detections:
        if det.bbox is None:
            continue

        bbox = tuple(det.bbox)

        if bbox_format == "xywh":
            x, y, bw, bh = bbox
            x1 = int(round(x))
            y1 = int(round(y))
            x2 = int(round(x + bw))
            y2 = int(round(y + bh))

        elif bbox_format == "xyxy":
            x1, y1, x2, y2 = map(lambda v: int(round(v)), bbox)

        else:
            raise ValueError(f"Unknown bbox_format: {bbox_format}")

        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        label = getattr(det, "label", "object")
        object_idx = getattr(det, "object_idx", "NA")
        text = f"{label}:{object_idx}"

        cv2.rectangle(vis, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)
        draw_label(vis, text, x1, y1)

    return vis

def draw_bbox_predictions_on_image(output_root, original_img, frame_det, bbox_format="xywh"):
    bbox_vis_path = output_root / "bbox_output.png"
    bbox_frame = draw_detections_on_frame(
        original_img,
        frame_det.objects,
        bbox_format=bbox_format
    )
    cv2.imwrite(str(bbox_vis_path), bbox_frame)

def create_writer_for_bbox_drawing(output_root, fps, width, height):
    bbox_vis_path = output_root / "bbox_output.mp4"
    bbox_writer = cv2.VideoWriter(
        str(bbox_vis_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not bbox_writer.isOpened():
        bbox_writer.release()
        raise RuntimeError(f"Failed to open bbox output video writer: {bbox_vis_path}")
    return bbox_writer

def draw_bboxes_on_video_frame_and_save(bbox_writer, original_frame, frame_det, bbox_format="xywh"):
    bbox_frame = draw_detections_on_frame(
        original_frame,
        frame_det.objects,
        bbox_format=bbox_format)
    
    bbox_writer.write(bbox_frame)