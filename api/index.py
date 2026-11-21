"""Stateless Vercel API. Vercel Blob holds records and private evidence objects."""
import base64
import os
import re
import tempfile
import time
import uuid
import logging
import json
import threading
from pathlib import Path
from urllib.request import urlopen

from backend import storage
from backend.detection import summarize_sequence
from backend.evidence import finalize_recording
from fastapi import FastAPI, HTTPException, Request, Form, File, UploadFile
from fastapi.responses import JSONResponse, Response

os.environ.setdefault('YOLO_CONFIG_DIR', os.path.join(tempfile.gettempdir(), 'acciden-yolo'))
os.environ.setdefault('MPLCONFIGDIR', os.path.join(tempfile.gettempdir(), 'acciden-matplotlib'))

app = FastAPI()
model = None
model_lock = threading.Lock()


@app.middleware('http')
async def no_cache(request: Request, call_next):
    try:
        response = await call_next(request)
    except Exception:
        logging.getLogger('acciden').exception('API request failed: %s', request.url.path)
        response = JSONResponse({'detail': 'The backend could not complete this request. Check the Vercel runtime logs for the cause.'}, status_code=500)
    response.headers['Cache-Control'] = 'no-store'
    return response


def record(incident_id):
    if not re.fullmatch(r'INC-[A-Za-z0-9-]+', incident_id):
        raise HTTPException(400, 'Invalid incident ID')
    return storage.get(incident_id)


def update(row, changes):
    return storage.update(row, changes)


def video_url(incident_id):
    return f'/api/video/{incident_id}'


@app.get('/api/status')
def status():
    return {'status': 'online', 'storage': 'vercel-blob', 'model_loaded': model is not None,
            'inference_size': 512, 'confirmation': '2 of 3 matching frames'}


@app.get('/api/response/{agency}/incidents')
def incidents(agency: str):
    if agency not in ('ambulance', 'police'):
        raise HTTPException(404, 'Unknown agency')
    rows = sorted(storage.rows(), key=lambda row: row['payload'].get('timestamp', ''), reverse=True)
    items = []
    for row in rows:
        item = row['payload']
        if item.get('has_clip') and item.get('object_path'):
            item['clip_url'] = video_url(item['id'])
        items.append(item)
    return {'success': True, 'incidents': items}


def get_detection_model():
    global model
    if model is None:
        from ultralytics import YOLO
        import torch
        torch.set_num_threads(1)
        target = Path(tempfile.gettempdir()) / 'acciden-epoch61.pt'
        if not target.exists():
            url = os.getenv('MODEL_URL', '')
            if not url.startswith('https://'):
                raise HTTPException(503, 'Set MODEL_URL to a downloadable HTTPS model file')
            # Unique temporary download avoids exposing a partially downloaded model.
            temporary = target.with_name(uuid.uuid4().hex + '.pt')
            try:
                from urllib.request import Request as DownloadRequest
                from urllib.parse import urlparse
                hostname = urlparse(url).hostname or ''
                headers = {}
                if hostname.endswith('.private.blob.vercel-storage.com'):
                    headers['Authorization'] = 'Bearer ' + os.getenv('BLOB_READ_WRITE_TOKEN', '')
                with urlopen(DownloadRequest(url, headers=headers), timeout=90) as source, temporary.open('wb') as destination:
                    import shutil
                    shutil.copyfileobj(source, destination)
                if temporary.stat().st_size < 1000000:
                    raise HTTPException(503, 'MODEL_URL did not return the model weights')
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
        model = YOLO(str(target))
    return model


@app.post('/api/detect')
def detect(base64_image: str = Form(''), frames_json: str = Form(''),
           conf: float = Form(.5), location: str = Form('Location not provided'),
           preview_only: bool = Form(False)):
    import cv2
    import numpy as np
    try:
        encoded_frames = json.loads(frames_json) if frames_json else [base64_image]
    except (ValueError, TypeError):
        raise HTTPException(400, 'Invalid frame sequence')
    if not isinstance(encoded_frames, list) or len(encoded_frames) not in (1, 3):
        raise HTTPException(400, 'Send exactly three frames to confirm an accident')
    if any(not isinstance(frame, str) or not frame for frame in encoded_frames):
        raise HTTPException(400, 'Invalid image frame')
    if sum(map(len, encoded_frames)) > 2_800_000:
        raise HTTPException(413, 'Frame sequence too large; use a lower source resolution')
    frames = []
    for encoded in encoded_frames:
        try:
            decoded = base64.b64decode(encoded.split(',')[-1], validate=True)
            frame = cv2.imdecode(np.frombuffer(decoded, np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            raise HTTPException(400, 'Invalid image frame')
        if frame is None:
            raise HTTPException(400, 'Invalid image frame')
        frames.append(frame)
    if any(frame.shape != frames[0].shape for frame in frames):
        raise HTTPException(400, 'All samples must come from the same video source')
    with model_lock:
        detector = get_detection_model()
        results = list(detector.predict(frames, conf=max(.1, min(.9, conf)),
                                        imgsz=512, verbose=False, device='cpu'))
        sequence = []
        for result in results:
            boxes = []
            for box in result.boxes:
                label = detector.names[int(box.cls[0])].strip().lower()
                boxes.append({'class_name': label, 'bbox': box.xyxy[0].tolist(),
                              'confidence_pct': round(float(box.conf[0])*100, 1),
                              'is_accident': label == 'accident'})
            sequence.append(boxes)
    detections, summary = summarize_sequence(sequence)
    incident = None
    if summary['accident_detected'] and not preview_only:
        incident_id = 'INC-' + uuid.uuid4().hex
        payload = {'id': incident_id, 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()),
                   'location': location[:200], 'confidence': summary['max_accident_confidence'],
                   'has_clip': False, 'status': 'RECORDING_EVIDENCE', 'clip_url': None,
                   'responses': {}, 'confirmation': summary['confirmation_rule']}
        if storage.create(incident_id, payload):
            incident = payload
    return {'success': True, 'incident': incident, 'detections': detections, 'summary': summary}


@app.post('/api/clips/upload')
def upload(incident_id: str = Form(...), video_file: UploadFile = File(...)):
    row = record(incident_id)
    if row['payload'].get('has_clip') or row['payload'].get('video_deleted'):
        raise HTTPException(409, 'Evidence already uploaded or deleted')
    contents = video_file.file.read(3_000_001)
    if not 512 <= len(contents) <= 3_000_000:
        raise HTTPException(413, 'Recording must be between 512 bytes and 3 MB')
    suffix = '.mp4' if 'mp4' in (video_file.content_type or '') else '.webm'
    contents, duration = finalize_recording(contents, suffix)
    object_path = incident_id + '-' + uuid.uuid4().hex + '.mp4'
    storage.put_video(object_path, contents, 'video/mp4')
    try:
        update(row, {'has_clip': True, 'object_path': object_path, 'clip_bytes': len(contents),
                     'clip_duration': duration, 'status': 'AWAITING_CONFIRMATION'})
    except Exception:
        storage.delete_video(object_path)
        raise
    return {'success': True}


@app.post('/api/response/{agency}/{action}')
def respond(agency: str, action: str, incident_id: str = Form(...), notes: str = Form('')):
    if agency not in ('ambulance', 'police') or action not in ('confirm', 'reject', 'resolve'):
        raise HTTPException(404, 'Unknown action')
    row = record(incident_id)
    if not row['payload'].get('has_clip'):
        raise HTTPException(409, 'Playable evidence is required')
    responses = row['payload'].get('responses', {})
    previous = responses.get(agency, {}).get('status', 'PENDING')
    if (action == 'resolve' and previous != 'DISPATCHED') or (action != 'resolve' and previous != 'PENDING'):
        raise HTTPException(409, 'Response already changed')
    responses[agency] = {'status': {'confirm': 'DISPATCHED', 'reject': 'REJECTED', 'resolve': 'RESOLVED'}[action], 'notes': notes[:2000]}
    return {'success': True, 'incident': update(row, {'responses': responses})}


@app.post('/api/incidents/{incident_id}/recording-failed')
def failed(incident_id: str):
    row = record(incident_id)
    if not row['payload'].get('has_clip'):
        update(row, {'status': 'RECORDING_FAILED'})
    return {'success': True}


@app.delete('/api/incidents/{incident_id}/video')
def delete_video(incident_id: str):
    row = record(incident_id)
    if row['payload'].get('status') == 'RECORDING_EVIDENCE':
        raise HTTPException(409, 'Wait until recording finishes before deleting')
    path = row['payload'].get('object_path')
    # Hide evidence first. Keep object_path until deletion succeeds so retries work.
    update(row, {'has_clip': False, 'video_deleted': True, 'clip_url': None, 'status': 'EVIDENCE_DELETED'})
    if path:
        storage.delete_video(path)
    update(record(incident_id), {'object_path': None, 'clip_bytes': 0})
    return {'success': True}


@app.get('/api/video/{incident_id}')
def play_video(incident_id: str, request: Request):
    row = record(incident_id)
    if not row['payload'].get('has_clip') or not row['payload'].get('object_path'):
        raise HTTPException(404, 'Video deleted or unavailable')
    contents, content_type = storage.video(row['payload']['object_path'])
    size = len(contents)
    headers = {'Cache-Control': 'private, no-store', 'Accept-Ranges': 'bytes'}
    requested = request.headers.get('range')
    if requested:
        match = re.fullmatch(r'bytes=(\d*)-(\d*)', requested)
        if not match or not any(match.groups()):
            return Response(status_code=416, headers={'Content-Range': f'bytes */{size}'})
        first, last = match.groups()
        start = int(first) if first else max(0, size - int(last))
        end = min(int(last), size - 1) if first and last else size - 1
        if start > end or start >= size:
            return Response(status_code=416, headers={'Content-Range': f'bytes */{size}'})
        headers['Content-Range'] = f'bytes {start}-{end}/{size}'
        return Response(contents[start:end+1], status_code=206, media_type=content_type, headers=headers)
    return Response(contents, media_type=content_type, headers=headers)
