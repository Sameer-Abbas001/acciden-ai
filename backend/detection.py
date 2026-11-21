"""YOLO accident detection shared by the local and cloud servers."""

DEFAULT_CONFIDENCE = 0.25
DEFAULT_IMAGE_SIZE = 320


def detect_frame(model, frame, conf=DEFAULT_CONFIDENCE, imgsz=DEFAULT_IMAGE_SIZE, *, live_camera=False, **options):
    detections = _predict_boxes(model, frame, conf, imgsz, options)
    if live_camera and imgsz == DEFAULT_IMAGE_SIZE and not detections:
        imgsz = 640
        detections = _predict_boxes(model, frame, conf, imgsz, options)
    return detections, imgsz


def _predict_boxes(model, frame, conf, imgsz, options):
    detections = []
    for result in model(frame, conf=conf, imgsz=imgsz, verbose=False, **options):
        for box in result.boxes:
            class_id = int(box.cls[0])
            name = model.names[class_id]
            score = float(box.conf[0])
            detections.append({
                'class_id': class_id, 'class_name': name, 'is_accident': name == 'accident',
                'confidence': score, 'confidence_pct': round(score * 100, 1),
                'bbox': [int(value) for value in box.xyxy[0]],
            })
    return detections


def summarize_frame(detections, conf, imgsz):
    accidents = [box for box in detections if box['is_accident']]
    peak = max((box['confidence_pct'] for box in accidents), default=0)
    return {
        'accident_detected': bool(accidents),
        'candidate_detected': bool(accidents),
        'max_accident_confidence': peak,
        'max_candidate_confidence': peak,
        'vehicles_count': len(detections) - len(accidents),
        'frames_analyzed': 1,
        'confirmation_rule': 'accident class detected in the current frame',
        'minimum_alert_confidence': round(conf * 100, 1),
        'inference_size': imgsz,
    }
