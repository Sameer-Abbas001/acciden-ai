import os
import io
import time
import base64
import json
import logging
import subprocess
import uuid
import asyncio
from contextlib import asynccontextmanager
from typing import Annotated, Optional, List
from collections import deque

import cv2
import numpy as np
from PIL import Image
import socketio
import imageio_ffmpeg

from fastapi import FastAPI, File, UploadFile, Query, Form, HTTPException, Body
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from backend.frames import dashboard_frame, decode_image
from backend.detection import DEFAULT_CONFIDENCE, DEFAULT_IMAGE_SIZE, detect_frame

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("acciden_server")

# Deployment configuration (local defaults preserve the desktop workflow).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.getenv("DATA_DIR", BASE_DIR))
os.makedirs(DATA_DIR, exist_ok=True)
ALLOWED_ORIGINS = [origin.strip() for origin in os.getenv(
    "ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000"
).split(",") if origin.strip()]

# ==============================================================================
# Socket.IO Real-Time Engine Setup
# ==============================================================================
sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins=ALLOWED_ORIGINS,
    logger=False,
    engineio_logger=False
)

# Global State & Model Configuration
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(BASE_DIR, "model", "epoch61.pt"))
DEMO_VIDEO_PATH = os.path.join(os.path.dirname(__file__), "static", "assets", "demo_accident.mp4")
if not os.path.exists(DEMO_VIDEO_PATH):
    DEMO_VIDEO_PATH = os.path.join(os.path.dirname(__file__), "runs", "detect", "predict", "accident.avi")

CLIPS_DIR = os.path.join(DATA_DIR, "clips") if "DATA_DIR" in os.environ else os.path.join(BASE_DIR, "static", "clips")
os.makedirs(CLIPS_DIR, exist_ok=True)

yolo_model = None
stats = {
    "start_time": time.time(),
    "total_inferences": 0,
    "accidents_detected": 0,
    "vehicles_detected": 0,
    "last_inference_ms": 0.0
}

# Continuous Real-Time Rolling Camera Buffers (240 frames @ 30fps = 8.0 seconds of actual camera feed)
recent_camera_frames = deque(maxlen=240)
recent_annotated_camera_frames = deque(maxlen=240)
last_server_accident_time = 0.0
active_clip_uploads = set()

# Ambulance Fleet Live Telemetry Database
ambulance_fleet = [
    {
        "id": "AMB-01",
        "callsign": "Alpha-1",
        "type": "Advanced Life Support (ALS)",
        "base": "Station 1 - Central District",
        "status": "STANDBY",  # STANDBY, EN_ROUTE, ON_SCENE, HOSPITAL_TRANSFER
        "crew": "Dr. Miller, Paramedic Ray",
        "equipment": ["Ventilator", "Defibrillator", "Oxygen", "Trauma Kit"],
        "eta_mins": 4,
        "lat": 28.6105,
        "lng": 77.2030,
        "fuel": "94%",
        "rating": "Level-1 Certified"
    },
    {
        "id": "AMB-02",
        "callsign": "Bravo-2",
        "type": "Rapid Response Trauma Unit",
        "base": "Station 2 - Northway Sector",
        "status": "STANDBY",
        "crew": "Paramedic Diaz, EMT Chen",
        "equipment": ["Defibrillator", "Extrication Gear", "Blood Kit"],
        "eta_mins": 7,
        "lat": 28.6270,
        "lng": 77.2210,
        "fuel": "88%",
        "rating": "Critical Rescue"
    },
    {
        "id": "AMB-03",
        "callsign": "Charlie-3",
        "type": "Critical Pediatric & Mobile ICU",
        "base": "Metro Trauma Center",
        "status": "STANDBY",
        "crew": "Dr. Sarah Khan, EMT Watson",
        "equipment": ["Mobile ICU", "Incubator", "Ventilator"],
        "eta_mins": 6,
        "lat": 28.6010,
        "lng": 77.1950,
        "fuel": "92%",
        "rating": "Mobile ICU"
    },
    {
        "id": "AMB-04",
        "callsign": "Delta Interceptor",
        "type": "Highway Express ALS Interceptor",
        "base": "Highway Sector 7 Outpost",
        "status": "STANDBY",
        "crew": "Lead Paramedic Vance, EMT Brooks",
        "equipment": ["ALS Trauma Suite", "Spinal Board", "Hydraulic Rescue"],
        "eta_mins": 3,
        "lat": 28.6160,
        "lng": 77.2120,
        "fuel": "96%",
        "rating": "High-Speed Interceptor"
    }
]

# Trauma Centers & Emergency Hospitals
hospitals = [
    {
        "id": "HOSP-01",
        "name": "Metro General Level-1 Trauma Center",
        "distance_km": 2.4,
        "eta_mins": 5,
        "icu_beds_free": 6,
        "er_status": "NORMAL CAPACITY",
        "blood_bank": "AVAILABLE",
        "contact": "+1 (800) 555-0199",
        "lat": 28.6180,
        "lng": 77.2010
    },
    {
        "id": "HOSP-02",
        "name": "St. Jude Memorial Emergency Hospital",
        "distance_km": 3.8,
        "eta_mins": 8,
        "icu_beds_free": 4,
        "er_status": "ACCEPTING TRAUMA",
        "blood_bank": "AVAILABLE",
        "contact": "+1 (800) 555-0144",
        "lat": 28.6290,
        "lng": 77.2150
    },
    {
        "id": "HOSP-03",
        "name": "Highway Express Emergency Post",
        "distance_km": 1.2,
        "eta_mins": 3,
        "icu_beds_free": 2,
        "er_status": "STABILIZATION ONLY",
        "blood_bank": "LIMITED",
        "contact": "+1 (800) 555-0112",
        "lat": 28.6145,
        "lng": 77.2085
    }
]

# Persistent incident records and shared forensic evidence
STATE_FILE = os.path.join(DATA_DIR, "incidents.json")
try:
    with open(STATE_FILE, encoding="utf-8") as state_file:
        incidents_db = json.load(state_file)
except (OSError, ValueError):
    incidents_db = []  # Never present sample emergencies as live incidents.

# Recover existing evidence without inventing detection scores or locations.
known_ids = {item["id"] for item in incidents_db}
for entry in os.scandir(CLIPS_DIR):
    if entry.is_file() and entry.name.startswith("clip_") and entry.name.endswith(".mp4") and ".tmp." not in entry.name and entry.stat().st_size >= 1000:
        saved_id = entry.name[5:-4]
        if saved_id not in known_ids:
            incidents_db.append({
                "id": saved_id, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(entry.stat().st_mtime)),
                "location": "Saved recording · location unavailable", "confidence": None,
                "status": "AWAITING_CONFIRMATION", "has_clip": True,
                "clip_url": f"/static/clips/{entry.name}", "clip_duration": 8.0,
            })
for incident in incidents_db:
    if incident.get("status") == "RECORDING_EVIDENCE" and not incident.get("has_clip"):
        incident["status"] = "RECORDING_FAILED"


def save_incidents():
    temporary = STATE_FILE + ".tmp"
    with open(temporary, "w", encoding="utf-8") as state_file:
        json.dump(incidents_db, state_file, ensure_ascii=False, indent=2)
    os.replace(temporary, STATE_FILE)


def generate_8s_incident_clip(incident_id: str, custom_frames: Optional[List[np.ndarray]] = None) -> str:
    """
    Renders an 8-second (240 frames @ 30fps) H.264 MP4 video clip.
    Runs the YOLO model on each source frame and draws ACCIDENT/Vehicle
    bounding boxes so the ambulance dashboard clip shows actual detection.
    """
    out_file = os.path.join(CLIPS_DIR, f"clip_{incident_id}.mp4")
    web_url = f"/static/clips/clip_{incident_id}.mp4"

    # If already generated and valid, return it unless custom frames are provided
    if not custom_frames and os.path.exists(out_file) and os.path.getsize(out_file) > 10000:
        return web_url

    raw_frames = []
    # 1. Use custom frames if provided (already annotated from camera)
    already_annotated = False
    if custom_frames and len(custom_frames) > 0:
        raw_frames = custom_frames
        already_annotated = True  # Custom frames from test.py already have boxes drawn
    # 2. Or use recent annotated live camera frames
    elif len(recent_annotated_camera_frames) > 10:
        raw_frames = list(recent_annotated_camera_frames)
        already_annotated = True
    # 3. Or use raw live camera frames
    elif len(recent_camera_frames) > 10:
        raw_frames = list(recent_camera_frames)
    else:
        logger.warning("No real camera frames available for %s; refusing to generate a synthetic evidence clip", incident_id)
        return ""

    # Run YOLO model on each raw frame and draw detection bounding boxes
    model = get_model()
    annotated_frames = []
    for f_raw in raw_frames:
        if already_annotated:
            annotated_frames.append(f_raw.copy())
        else:
            ann = f_raw.copy()
            results = model(f_raw, conf=0.25, imgsz=320, verbose=False)
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = model.names.get(cls_id, f"class_{cls_id}")
                    confidence = float(box.conf[0])
                    pct = round(confidence * 100, 1)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    h_f, w_f = ann.shape[:2]
                    x1 = max(0, min(w_f - 1, x1))
                    y1 = max(0, min(h_f - 1, y1))
                    x2 = max(0, min(w_f - 1, x2))
                    y2 = max(0, min(h_f - 1, y2))

                    if cls_name.lower() == "accident":
                        color = (0, 0, 255)
                        label = f"ACCIDENT {pct}%"
                        cv2.rectangle(ann, (x1, y1), (x2, y2), color, 3)
                        # Corner reticles
                        clen = min(20, (x2 - x1) // 3)
                        cv2.line(ann, (x1, y1), (x1 + clen, y1), (0, 255, 255), 4)
                        cv2.line(ann, (x1, y1), (x1, y1 + clen), (0, 255, 255), 4)
                        cv2.line(ann, (x2, y1), (x2 - clen, y1), (0, 255, 255), 4)
                        cv2.line(ann, (x2, y1), (x2, y1 + clen), (0, 255, 255), 4)
                        cv2.line(ann, (x1, y2), (x1 + clen, y2), (0, 255, 255), 4)
                        cv2.line(ann, (x1, y2), (x1, y2 - clen), (0, 255, 255), 4)
                        cv2.line(ann, (x2, y2), (x2 - clen, y2), (0, 255, 255), 4)
                        cv2.line(ann, (x2, y2), (x2, y2 - clen), (0, 255, 255), 4)
                        cv2.putText(ann, label, (x1, max(30, y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    else:
                        color = (0, 255, 0)
                        label = f"Vehicle {pct}%"
                        cv2.rectangle(ann, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(ann, label, (x1, max(25, y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            annotated_frames.append(ann)

    target_fps = 30
    total_frames = 8 * target_fps  # 240 frames = 8.0 seconds
    clip_frames = []

    # Build 8-second clip: loop source frames if fewer than 240
    n_source = len(annotated_frames)
    for i in range(total_frames):
        if i < n_source:
            f = annotated_frames[i].copy()
        else:
            # Loop back around the source frames (replay the accident)
            f = annotated_frames[i % n_source].copy()
        h, w = f.shape[:2]

        # Minimal top HUD bar - pure ASCII only (no Unicode bullets)
        secs = i // target_fps
        millis = int((i % target_fps) * (1000.0 / target_fps))
        overlay = f.copy()
        top_h = max(32, int(h * 0.07))
        cv2.rectangle(overlay, (0, 0), (w, top_h), (10, 15, 25), -1)
        cv2.addWeighted(overlay, 0.7, f, 0.3, 0, f)

        font_scale = max(0.38, min(0.5, w / 750.0))
        # Pulsing REC dot
        dot_color = (0, 0, 255) if (i // 8) % 2 == 0 else (0, 140, 255)
        cv2.circle(f, (16, int(top_h * 0.55)), 6, dot_color, -1)
        cv2.putText(f, f"REC | {incident_id} | 00:{secs:02d}.{millis:03d} | 8.0s 1.0X",
                    (30, int(top_h * 0.7)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 240, 255), 1, cv2.LINE_AA)

        clip_frames.append(f)

    # Encode with imageio_ffmpeg to H.264 MP4 with faststart at 30 FPS
    try:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        h, w = clip_frames[0].shape[:2]
        cmd = [
            ffmpeg_exe,
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{w}x{h}",
            "-pix_fmt", "bgr24",
            "-r", str(target_fps),
            "-i", "-",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-profile:v", "high",
            "-level", "4.0",
            "-preset", "fast",
            "-crf", "22",
            "-movflags", "+faststart",
            "-an",
            out_file
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for f in clip_frames:
            proc.stdin.write(f.tobytes())
        proc.stdin.close()
        proc.wait()
        logger.info(f"Generated 8s clip with YOLO detections: {out_file} ({os.path.getsize(out_file)} bytes)")
        return web_url
    except Exception as e:
        logger.error(f"Error encoding 8s camera clip for {incident_id}: {e}")
        return ""

generate_10s_incident_clip = generate_8s_incident_clip  # Backwards compatibility alias


# Model Loader (UNALTERED)
def get_model():
    global yolo_model
    if yolo_model is None:
        logger.info(f"Loading YOLO model from: {MODEL_PATH}")
        yolo_model = YOLO(MODEL_PATH)
        logger.info(f"Model loaded successfully. Classes: {yolo_model.names}")
    return yolo_model


# ==============================================================================
# FastAPI Lifespan & Application Setup
# ==============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern FastAPI lifespan handler for startup/shutdown."""
    try:
        model = get_model()
        dummy_img = np.zeros((320, 320, 3), dtype=np.uint8)
        _ = model(dummy_img, imgsz=320, verbose=False)
        logger.info("Model warm-up completed successfully.")

        # Do not pre-render synthetic/demo evidence. Only actual camera recordings are playable.
        logger.info("Startup complete; waiting for real incident camera recordings.")
    except Exception as e:
        logger.error(f"Error initializing server during lifespan: {e}")
    yield
    logger.info("AcciDEN AI server shutting down.")


app = FastAPI(
    title="AcciDEN AI - Collision & Traffic Intelligence Hub",
    description="Real-Time Accident and Vehicle Detection API powered by YOLOv8 with Real-Time Ambulance Command Hub",
    version="2.6.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# Socket.IO Event Handlers
# ==============================================================================
@sio.on("connect")
async def handle_connect(sid, environ):
    logger.info(f"Socket.IO client connected: {sid}")
    await sio.emit("system_status", {
        "status": "ONLINE",
        "incidents_count": len(incidents_db),
        "fleet_count": len(ambulance_fleet),
        "active_incident": incidents_db[0] if incidents_db else None
    }, to=sid)


@sio.on("disconnect")
async def handle_disconnect(sid):
    logger.info(f"Socket.IO client disconnected: {sid}")


@sio.on("accident_alert")
async def handle_accident_alert(sid, data):
    """Persist a new client-detected incident before broadcasting it.

    The browser starts recording before a clip exists. Never publish its
    placeholder clip URL as if the MP4 were already saved.
    """
    incident = dict(data or {})
    incident_id = str(incident.get("id", "")).strip()
    if not incident_id:
        logger.warning("Ignoring accident_alert without an incident ID")
        return

    incident["clip_url"] = None
    incident["has_clip"] = False
    incident["clip_duration"] = 8.0
    incident["status"] = "RECORDING_EVIDENCE"

    existing = next((item for item in incidents_db if item.get("id") == incident_id), None)
    if existing:
        existing.update(incident)
        incident = existing
    else:
        incidents_db.insert(0, incident)
        del incidents_db[50:]

    logger.info("Accident alert received from %s: %s (waiting for clip upload)", sid, incident_id)
    await sio.emit("accident_detected", incident)


@sio.on("operator_confirm")
async def handle_socket_operator_confirm(sid, data):
    """Operator confirms accident is REAL and dispatches fleet."""
    incident_id = data.get("incident_id")
    unit_id = data.get("unit_id", "AMB-04")
    hospital_id = data.get("hospital_id", "HOSP-01")
    notes = data.get("notes", "Operator confirmed real collision from 8s forensic clip.")

    logger.info(f"Operator CONFIRMED accident {incident_id} -> Unit: {unit_id}")

    target_inc = None
    for inc in incidents_db:
        if inc["id"] == incident_id:
            inc["status"] = "CONFIRMED & DISPATCHED"
            inc["assigned_unit"] = unit_id
            inc["destination_hospital"] = hospital_id
            inc["operator_notes"] = notes
            inc["confirmed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            target_inc = inc
            break

    for unit in ambulance_fleet:
        if unit["id"] == unit_id:
            unit["status"] = "EN_ROUTE"
            break

    confirmation_payload = {
        "incident_id": incident_id,
        "status": "CONFIRMED & DISPATCHED",
        "unit_id": unit_id,
        "hospital_id": hospital_id,
        "eta_minutes": 4,
        "confirmed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "incident": target_inc
    }

    await sio.emit("incident_confirmed", confirmation_payload)
    await sio.emit("fleet_update", {"fleet": ambulance_fleet})


@sio.on("operator_reject")
async def handle_socket_operator_reject(sid, data):
    """Operator marks incident as False Alarm."""
    incident_id = data.get("incident_id")
    reason = data.get("reason", "Operator reviewed 8s clip: No real vehicular damage found.")

    logger.info(f"Operator REJECTED accident {incident_id} -> Reason: {reason}")

    target_inc = None
    for inc in incidents_db:
        if inc["id"] == incident_id:
            inc["status"] = "REJECTED (FALSE POSITIVE)"
            inc["operator_notes"] = reason
            target_inc = inc
            break

    reject_payload = {
        "incident_id": incident_id,
        "status": "REJECTED (FALSE POSITIVE)",
        "reason": reason,
        "incident": target_inc
    }

    await sio.emit("incident_rejected", reject_payload)


@sio.on("ping_hub")
async def handle_ping(sid, data):
    await sio.emit("pong_hub", {"time": time.time()}, to=sid)


# ==============================================================================
# Model Detection Processing
# ==============================================================================
def process_cv2_frame(frame: np.ndarray, conf: float = DEFAULT_CONFIDENCE, iou: Optional[float] = None, imgsz: int = DEFAULT_IMAGE_SIZE, *, live_camera: bool = False):
    model = get_model()
    start_t = time.time()
    h, w = frame.shape[:2]

    options = {} if iou is None else {'iou': iou}
    detections, used_size = detect_frame(model, frame, conf, imgsz, live_camera=live_camera, **options)
    inf_time_ms = (time.time() - start_t) * 1000

    accidents = [box for box in detections if box['is_accident']]
    accident_count = len(accidents)
    vehicle_count = len(detections) - accident_count
    accident_found = bool(accidents)
    max_acc_conf = max((box['confidence'] for box in accidents), default=0.0)
    for box in detections:
        x1, y1, x2, y2 = box['bbox']
        x1, x2 = (max(0, min(w - 1, x)) for x in (x1, x2))
        y1, y2 = (max(0, min(h - 1, y)) for y in (y1, y2))
        box['bbox'] = [x1, y1, x2, y2]
        box['bbox_norm'] = [round(x1 / w, 4), round(y1 / h, 4), round(x2 / w, 4), round(y2 / h, 4)]
        box['severity'] = ("CRITICAL" if box['confidence'] > 0.5 else "WARNING") if box['is_accident'] else "NORMAL"
        box['confidence'] = round(box['confidence'], 4)

    # Update system stats
    stats["total_inferences"] += 1
    stats["accidents_detected"] += accident_count
    stats["vehicles_detected"] += vehicle_count
    stats["last_inference_ms"] = inf_time_ms

    summary = {
        "accident_detected": accident_found,
        "max_accident_confidence": round(max_acc_conf * 100, 1),
        "total_detections": len(detections),
        "accidents_count": accident_count,
        "vehicles_count": vehicle_count,
        "severity_level": "LEVEL-1 CRITICAL" if accident_found and max_acc_conf >= 0.5 else ("LEVEL-2 WARNING" if accident_found else "ALL CLEAR"),
        "inference_size": used_size,
        "frame_width": w,
        "frame_height": h,
        "inference_ms": round(inf_time_ms, 2)
    }

    # Store raw live camera frame into buffer
    try:
        recent_camera_frames.append(frame.copy())
    except Exception:
        pass

    return detections, summary


# ==============================================================================
# REST API Endpoints
# ==============================================================================
@app.get("/api/status")
async def get_status():
    model = get_model()
    device = "CUDA (GPU)" if hasattr(model, "device") and "cuda" in str(model.device).lower() else "CPU"
    return {
        "status": "online",
        "model_file": os.path.basename(MODEL_PATH),
        "classes": model.names,
        "device": device,
        "uptime_seconds": round(time.time() - stats["start_time"], 1),
        "total_inferences": stats["total_inferences"],
        "accidents_detected": stats["accidents_detected"],
        "vehicles_detected": stats["vehicles_detected"],
        "last_inference_ms": round(stats["last_inference_ms"], 2)
    }


@app.get("/api/build-info")
async def build_info():
    return {"build": "camera-frame-2", "folder": BASE_DIR}


@app.post("/api/detect")
async def detect_api(
    file: Optional[UploadFile] = File(None),
    base64_image: Optional[str] = Form(None),
    conf: float = Form(DEFAULT_CONFIDENCE),
    iou: Optional[float] = Form(None),
    imgsz: int = Form(DEFAULT_IMAGE_SIZE),
    return_annotated: bool = Form(True),
    location: str = Form("Location not provided"),
    preview_only: Annotated[bool, Form()] = False,
    frames_json: Annotated[str, Form()] = "",
    live_camera: Annotated[bool, Form()] = False
):
    try:
        if file is not None:
            frame = decode_image(await file.read())
        else:
            frame = dashboard_frame(base64_image, frames_json)

        detections, summary = process_cv2_frame(frame, conf=conf, iou=iou, imgsz=imgsz, live_camera=live_camera)

        annotated_frame = frame.copy()
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            if d["is_accident"]:
                color = (0, 0, 255)  # Red for accident
                label = f"ACCIDENT {d['confidence_pct']}%"
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(annotated_frame, label, (x1, max(30, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            else:
                color = (255, 200, 0)
                label = f"Vehicle {d['confidence_pct']}%"
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, label, (x1, max(25, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # Store annotated live camera frame into buffer
        try:
            recent_annotated_camera_frames.append(annotated_frame.copy())
        except Exception:
            pass

        annotated_b64 = None
        if return_annotated:
            _, buf = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            annotated_b64 = f"data:image/jpeg;base64,{base64.b64encode(buf).decode('utf-8')}"

        # If accident detected, record real-time camera video clip!
        incident_record = None
        global last_server_accident_time
        now_ts = time.time()
        if summary["accident_detected"] and not preview_only and (now_ts - last_server_accident_time > 12.0):
            last_server_accident_time = now_ts
            incident_id = f"INC-{uuid.uuid4().hex[:12].upper()}"

            incident_record = {
                "id": incident_id,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "confidence": summary["max_accident_confidence"],
                "severity": summary["severity_level"],
                "vehicles_involved": summary["vehicles_count"],
                "status": "RECORDING_EVIDENCE",
                "location": location[:200],
                "coordinates": {"lat": 28.6139, "lng": 77.2090},
                "snapshot": "/static/assets/traffic_collision.jpg",
                "clip_url": None,
                "has_clip": False,
                "clip_duration": 8.0,
                "assigned_unit": None,
                "destination_hospital": None,
                "eta_minutes": 4,
                "operator_notes": None,
                "confirmed_at": None,
                "triage_summary": f"Live Screen Detection ({summary['max_accident_confidence']}% conf). Recording 8s Screen Video at 1.0x..."
            }
            incidents_db.insert(0, incident_record)
            if len(incidents_db) > 50:
                incidents_db.pop()

            # Broadcast live alert to ambulance dashboard via Socket.IO
            save_incidents()
            await sio.emit("accident_detected", incident_record)

        return {
            "success": True,
            "incident": incident_record,
            "summary": summary,
            "detections": detections,
            "annotated_image": annotated_b64
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in /api/detect")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/clips/upload")
async def upload_camera_clip(
    incident_id: str = Form(...),
    video_file: UploadFile = File(...),
    confidence: Optional[float] = Form(88.4),
    location: Optional[str] = Form("Sector 7 - Junction Express"),
    vehicles_count: Optional[int] = Form(2)
):
    """Save a browser recording and transcode it to a real H.264 MP4.

    Never rename a WebM file to .mp4: if FFmpeg fails, return an error and
    leave the incident marked as having no playable clip.
    """
    import re
    safe_incident_id = re.sub(r"[^A-Za-z0-9_-]", "", incident_id)
    if not safe_incident_id:
        raise HTTPException(status_code=400, detail="Invalid incident ID")

    raw_path = None
    temp_mp4 = os.path.join(CLIPS_DIR, f"clip_{safe_incident_id}.tmp.mp4")
    out_mp4 = os.path.join(CLIPS_DIR, f"clip_{safe_incident_id}.mp4")
    web_url = f"/static/clips/clip_{safe_incident_id}.mp4"

    if safe_incident_id in active_clip_uploads:
        raise HTTPException(503, "This recording is already being processed. Retry shortly.")
    active_clip_uploads.add(safe_incident_id)
    try:
        contents = await video_file.read()
        if not contents or len(contents) < 512:
            raise HTTPException(status_code=400, detail="The browser recording is empty or too small")

        existing = next((inc for inc in incidents_db if inc["id"] == safe_incident_id), None)
        if existing and existing.get("video_deleted"):
            raise HTTPException(409, "Evidence deleted")
        if existing and existing.get("has_clip") and os.path.isfile(out_mp4):
            return {"success": True, "clip_url": web_url, "size_bytes": os.path.getsize(out_mp4)}

        # Keep the original recording with the correct container extension.
        content_type = (video_file.content_type or "").lower()
        original_name = (video_file.filename or "").lower()
        if "mp4" in content_type or original_name.endswith(".mp4"):
            raw_ext = ".mp4"
        elif "ogg" in content_type or original_name.endswith(".ogv"):
            raw_ext = ".ogv"
        else:
            raw_ext = ".webm"
        raw_path = os.path.join(CLIPS_DIR, f"raw_{safe_incident_id}{raw_ext}")
        with open(raw_path, "wb") as f:
            f.write(contents)

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_exe, "-y", "-hide_banner", "-loglevel", "error",
            "-fflags", "+genpts", "-i", raw_path,
            "-map", "0:v:0", "-an", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,fps=30",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-profile:v", "main", "-preset", "fast", "-crf", "22",
            "-movflags", "+faststart", "-t", "8", temp_mp4
        ]
        res = await asyncio.to_thread(subprocess.run, cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0 or not os.path.exists(temp_mp4) or os.path.getsize(temp_mp4) < 1000:
            error_text = (res.stderr or "FFmpeg did not create a valid MP4")[-1200:]
            logger.error("Clip conversion failed for %s: %s", safe_incident_id, error_text)
            if os.path.exists(temp_mp4):
                os.remove(temp_mp4)
            raise HTTPException(status_code=422, detail=f"Could not convert browser recording to MP4: {error_text}")

        probe = cv2.VideoCapture(temp_mp4)
        fps = probe.get(cv2.CAP_PROP_FPS)
        duration = probe.get(cv2.CAP_PROP_FRAME_COUNT) / fps if fps else 0
        probe.release()
        if not 7.75 <= duration <= 8.1:
            raise HTTPException(status_code=422, detail="Evidence must contain a complete 8-second recording. The source may have ended early.")

        # Publish only after successful conversion so the dashboard never sees a partial file.
        os.replace(temp_mp4, out_mp4)

        found_inc = False
        for inc in incidents_db:
            if inc["id"] == safe_incident_id:
                inc["clip_url"] = web_url
                inc["has_clip"] = True
                inc["clip_duration"] = 8.0
                inc["status"] = "AWAITING_CONFIRMATION"
                found_inc = True
                break

        if not found_inc:
            new_inc = {
                "id": safe_incident_id,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "confidence": float(confidence or 88.4),
                "severity": "LEVEL-1 CRITICAL",
                "vehicles_involved": int(vehicles_count or 2),
                "status": "AWAITING_CONFIRMATION",
                "location": location or "Sector 7 - Junction Express",
                "coordinates": {"lat": 28.6139, "lng": 77.2090},
                "snapshot": "/static/assets/traffic_collision.jpg",
                "clip_url": web_url,
                "has_clip": True,
                "clip_duration": 8.0,
                "assigned_unit": None,
                "destination_hospital": None,
                "eta_minutes": 4,
                "operator_notes": None,
                "confirmed_at": None,
                "triage_summary": f"Live screen recording ({confidence or 88.4}% confidence). Operator forensic review required."
            }
            incidents_db.insert(0, new_inc)
            if len(incidents_db) > 50:
                incidents_db.pop()
            await sio.emit("accident_detected", new_inc)

        save_incidents()
        await sio.emit("clip_ready", {
            "incident_id": safe_incident_id,
            "clip_url": web_url,
            "duration": 8.0,
            "source": "SCREEN_RECORDING"
        })
        logger.info("Saved playable MP4 for %s (%s bytes)", safe_incident_id, os.path.getsize(out_mp4))
        return {"success": True, "clip_url": web_url, "size_bytes": os.path.getsize(out_mp4)}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in /api/clips/upload")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        active_clip_uploads.discard(safe_incident_id)
        if os.path.exists(temp_mp4):
            try:
                os.remove(temp_mp4)
            except OSError:
                pass


def generate_video_stream(video_path: str, conf: float = 0.25):
    while True:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video stream at: {video_path}")
            break

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_delay = 1.0 / max(10.0, min(fps, 30.0))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t0 = time.time()
            detections, summary = process_cv2_frame(frame, conf=conf, imgsz=320)

            annotated = frame.copy()
            h, w = annotated.shape[:2]

            overlay = annotated.copy()
            cv2.rectangle(overlay, (0, 0), (w, 40), (10, 15, 25), -1)
            cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)

            hud_title = f"AI VISION HUD | INF: {summary['inference_ms']}ms | STAT: {summary['severity_level']}"
            cv2.putText(annotated, hud_title, (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 50, 255) if summary["accident_detected"] else (0, 255, 180), 2)

            for d in detections:
                x1, y1, x2, y2 = d["bbox"]
                if d["is_accident"]:
                    color = (0, 0, 255)
                    label = f"ACCIDENT {d['confidence_pct']}%"
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
                    corner_len = min(20, (x2 - x1) // 3)
                    cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), (0, 255, 255), 4)
                    cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), (0, 255, 255), 4)
                    cv2.putText(annotated, label, (x1, max(30, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                else:
                    color = (255, 180, 0)
                    label = f"Vehicle {d['confidence_pct']}%"
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated, label, (x1, max(25, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            _, buf = cv2.imencode('.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            frame_bytes = buf.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            elapsed = time.time() - t0
            sleep_time = max(0.01, frame_delay - elapsed)
            time.sleep(sleep_time)

        cap.release()


@app.get("/api/stream/demo")
async def stream_demo(conf: float = Query(0.25)):
    if not os.path.exists(DEMO_VIDEO_PATH):
        raise HTTPException(status_code=404, detail="Demo video not found")
    return StreamingResponse(
        generate_video_stream(DEMO_VIDEO_PATH, conf=conf),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


def generate_webcam_stream(cam_idx: int = 0, conf: float = 0.25):
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        logger.error(f"Cannot open webcam index {cam_idx}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        detections, summary = process_cv2_frame(frame, conf=conf, imgsz=320)
        annotated = frame.copy()

        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            if d["is_accident"]:
                color = (0, 0, 255)
                label = f"ACCIDENT {d['confidence_pct']}%"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
                cv2.putText(annotated, label, (x1, max(30, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            else:
                color = (0, 255, 120)
                label = f"Vehicle {d['confidence_pct']}%"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, label, (x1, max(25, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        _, buf = cv2.imencode('.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        frame_bytes = buf.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)

    cap.release()


@app.get("/api/stream/webcam")
async def stream_webcam(conf: float = Query(0.25)):
    return StreamingResponse(
        generate_webcam_stream(cam_idx=0, conf=conf),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/incidents")
async def get_incidents():
    return incidents_db


# ==============================================================================
# Ambulance Command Hub REST APIs
# ==============================================================================
@app.get("/api/ambulance/status")
async def get_ambulance_status():
    standby_units = sum(1 for u in ambulance_fleet if u["status"] == "STANDBY")
    enroute_units = sum(1 for u in ambulance_fleet if u["status"] == "EN_ROUTE")
    pending_incidents = sum(1 for i in incidents_db if i["status"] == "AWAITING_CONFIRMATION")
    confirmed_incidents = sum(1 for i in incidents_db if i["status"] == "CONFIRMED & DISPATCHED")

    return {
        "status": "OPERATIONAL",
        "system_link": "SOCKET_IO_SYNCED",
        "total_fleet": len(ambulance_fleet),
        "standby_units": standby_units,
        "enroute_units": enroute_units,
        "pending_verifications": pending_incidents,
        "confirmed_dispatches": confirmed_incidents,
        "avg_response_time_mins": 4.2,
        "hospitals_count": len(hospitals),
        "latest_incident": incidents_db[0] if incidents_db else None
    }


@app.get("/api/ambulance/incidents")
async def get_ambulance_incidents():
    return {
        "success": True,
        "count": len(incidents_db),
        "incidents": incidents_db
    }


@app.get("/api/ambulance/recent-videos")
async def get_recent_accident_videos():
    """Return playable MP4 recordings found on disk, newest first.

    This endpoint deliberately reads the clip directory instead of relying
    only on the in-memory incident list, so saved videos remain visible after
    page reloads and even if a Socket.IO event was missed.
    """
    incident_by_id = {str(item.get("id")): item for item in incidents_db if item.get("id")}
    videos = []
    try:
        for entry in os.scandir(CLIPS_DIR):
            if not entry.is_file() or not entry.name.lower().endswith(".mp4"):
                continue
            if not entry.name.startswith("clip_") or ".tmp." in entry.name:
                continue
            stat = entry.stat()
            if stat.st_size < 1000:
                continue
            incident_id = entry.name[len("clip_"):-len(".mp4")]
            incident = incident_by_id.get(incident_id, {})
            videos.append({
                "id": incident_id,
                "timestamp": incident.get("timestamp") or time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
                "confidence": incident.get("confidence", "—"),
                "severity": incident.get("severity", "ACCIDENT RECORDING"),
                "location": incident.get("location", "Location unavailable"),
                "status": incident.get("status", "VIDEO SAVED"),
                "clip_url": f"/static/clips/{entry.name}",
                "has_clip": True,
                "size_bytes": stat.st_size,
                "recorded_at": stat.st_mtime,
            })
    except OSError:
        logger.exception("Could not scan saved accident videos")
        raise HTTPException(status_code=500, detail="Could not read saved accident videos")

    videos.sort(key=lambda item: item["recorded_at"], reverse=True)
    return {"success": True, "count": len(videos), "videos": videos[:20]}


@app.get("/api/ambulance/incident/{incident_id}")
async def get_ambulance_incident(incident_id: str):
    for inc in incidents_db:
        if inc["id"] == incident_id:
            return {"success": True, "incident": inc}
    raise HTTPException(status_code=404, detail="Incident not found")


@app.post("/api/ambulance/confirm")
async def confirm_accident(
    incident_id: str = Form(...),
    unit_id: str = Form("AMB-04"),
    hospital_id: str = Form("HOSP-01"),
    notes: Optional[str] = Form("Operator confirmed genuine collision via 8-second forensic video playback.")
):
    target_inc = None
    for inc in incidents_db:
        if inc["id"] == incident_id:
            inc["status"] = "CONFIRMED & DISPATCHED"
            inc["assigned_unit"] = unit_id
            inc["destination_hospital"] = hospital_id
            inc["operator_notes"] = notes
            inc["confirmed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            target_inc = inc
            break

    if not target_inc:
        raise HTTPException(status_code=404, detail=f"Incident {incident_id} not found")

    for unit in ambulance_fleet:
        if unit["id"] == unit_id:
            unit["status"] = "EN_ROUTE"
            break

    confirmation_payload = {
        "incident_id": incident_id,
        "status": "CONFIRMED & DISPATCHED",
        "unit_id": unit_id,
        "hospital_id": hospital_id,
        "eta_minutes": 4,
        "confirmed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "incident": target_inc
    }

    await sio.emit("incident_confirmed", confirmation_payload)
    await sio.emit("fleet_update", {"fleet": ambulance_fleet})

    return {
        "success": True,
        "message": f"Accident {incident_id} successfully confirmed! Unit {unit_id} dispatched.",
        "confirmation": confirmation_payload
    }


@app.post("/api/ambulance/reject")
async def reject_accident(
    incident_id: str = Form(...),
    reason: Optional[str] = Form("8-second clip reviewed: No vehicular impact or injuries detected. Marked as False Alarm.")
):
    target_inc = None
    for inc in incidents_db:
        if inc["id"] == incident_id:
            inc["status"] = "REJECTED (FALSE POSITIVE)"
            inc["operator_notes"] = reason
            target_inc = inc
            break

    if not target_inc:
        raise HTTPException(status_code=404, detail=f"Incident {incident_id} not found")

    reject_payload = {
        "incident_id": incident_id,
        "status": "REJECTED (FALSE POSITIVE)",
        "reason": reason,
        "incident": target_inc
    }

    await sio.emit("incident_rejected", reject_payload)

    return {
        "success": True,
        "message": f"Accident {incident_id} marked as False Alarm.",
        "result": reject_payload
    }


@app.get("/api/ambulance/units")
async def get_ambulance_units():
    return {"success": True, "units": ambulance_fleet}


@app.post("/api/ambulance/units/{unit_id}/status")
async def update_unit_status(unit_id: str, status: str = Form(...)):
    for unit in ambulance_fleet:
        if unit["id"] == unit_id:
            unit["status"] = status
            await sio.emit("fleet_update", {"fleet": ambulance_fleet})
            return {"success": True, "unit": unit}
    raise HTTPException(status_code=404, detail="Unit not found")


@app.get("/api/ambulance/hospitals")
async def get_hospitals():
    return {"success": True, "hospitals": hospitals}


@app.post("/api/clips/generate")
async def trigger_clip_generation(incident_id: str = Form(...)):
    live_frames_snapshot = list(recent_annotated_camera_frames)
    clip_url = generate_8s_incident_clip(incident_id, custom_frames=live_frames_snapshot)
    if not clip_url:
        raise HTTPException(status_code=500, detail="Failed to generate 8s clip")

    for inc in incidents_db:
        if inc["id"] == incident_id:
            inc["clip_url"] = clip_url
            inc["has_clip"] = True
            inc["clip_duration"] = 8.0
            break

    await sio.emit("clip_ready", {
        "incident_id": incident_id,
        "clip_url": clip_url,
        "duration": 8.0,
        "source": "REAL_TIME_CAMERA"
    })

    return {"success": True, "clip_url": clip_url}


@app.post("/api/trigger_test_accident")
async def trigger_test_accident(
    location: str = Form("Sector 7 - Junction Express"),
    confidence: float = Form(88.4)
):
    """
    Simulation endpoint that creates an accident incident, generates
    the 8-second clip from real-time camera frames, and broadcasts to both dashboards.
    """
    new_id = f"INC-{int(time.time()*1000)%1000000}"
    live_frames_snapshot = list(recent_annotated_camera_frames)
    clip_url = generate_8s_incident_clip(new_id, custom_frames=live_frames_snapshot)

    incident_record = {
        "id": new_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "confidence": confidence,
        "severity": "LEVEL-1 CRITICAL",
        "vehicles_involved": 2,
        "status": "AWAITING_CONFIRMATION",
        "location": location,
        "coordinates": {"lat": 28.6139, "lng": 77.2090},
        "snapshot": "/static/assets/traffic_collision.jpg",
        "clip_url": clip_url,
        "has_clip": bool(clip_url),
        "clip_duration": 8.0,
        "assigned_unit": None,
        "destination_hospital": None,
        "eta_minutes": 4,
        "operator_notes": None,
        "confirmed_at": None,
        "triage_summary": f"High-impact collision in {location}. Real-time screen 8s video verification required."
    }
    incidents_db.insert(0, incident_record)
    if len(incidents_db) > 50:
        incidents_db.pop()

    await sio.emit("accident_detected", incident_record)
    if clip_url:
        await sio.emit("clip_ready", {
            "incident_id": new_id,
            "clip_url": clip_url,
            "duration": 8.0,
            "source": "REAL_TIME_CAMERA"
        })

    return {
        "success": True,
        "message": f"Test accident incident {new_id} generated.",
        "incident": incident_record
    }


@app.post("/api/dispatch")
async def trigger_dispatch(incident_id: str = Form("INC-849201"), agency: str = Form("EMS & Highway Patrol")):
    ticket_id = f"DISPATCH-{int(time.time())%100000}"
    return {
        "success": True,
        "dispatch_id": ticket_id,
        "incident_id": incident_id,
        "agency": agency,
        "units": ["Ambulance Unit AMB-04", "Rescue 3", "Traffic Interceptor 09"],
        "estimated_arrival_minutes": 4,
        "status": "EN_ROUTE",
        "timestamp": time.strftime("%H:%M:%S UTC")
    }


# ==============================================================================
# Shared evidence, independent agency decisions
# ==============================================================================
def check_agency(agency):
    if agency not in ("ambulance", "police"):
        raise HTTPException(status_code=404, detail="Unknown response agency")


@app.get("/api/response/{agency}/incidents")
async def response_incidents(agency: str):
    check_agency(agency)
    return {"success": True, "incidents": incidents_db}


@app.post("/api/response/{agency}/{action}")
async def response_action(agency: str, action: str, incident_id: str = Form(...),
                          notes: str = Form("")):
    check_agency(agency)
    if action not in ("confirm", "reject", "resolve"):
        raise HTTPException(status_code=404, detail="Unknown review action")
    incident = next((item for item in incidents_db if item["id"] == incident_id), None)
    if incident is None:
        raise HTTPException(status_code=404, detail="Incident not found")
    if not incident.get("has_clip"):
        raise HTTPException(status_code=409, detail="Wait for playable evidence before reviewing")
    responses = incident.setdefault("responses", {})
    previous = responses.get(agency, {}).get("status", "PENDING")
    if previous == "RESOLVED" or (action == "resolve" and previous != "DISPATCHED"):
        raise HTTPException(status_code=409, detail="This action is not available for the current response status")
    if action != "resolve" and previous != "PENDING":
        raise HTTPException(status_code=409, detail="This agency has already reviewed this incident")
    responses[agency] = {
        "status": {"confirm": "DISPATCHED", "reject": "REJECTED", "resolve": "RESOLVED"}[action],
        "notes": notes[:2000], "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dispatch_mode": "LOCAL_SIMULATION"
    }
    save_incidents()
    await sio.emit("agency_response", {"agency": agency, "incident": incident})
    return {"success": True, "incident": incident}


@app.post("/api/incidents/{incident_id}/recording-failed")
async def recording_failed(incident_id: str):
    incident = next((item for item in incidents_db if item["id"] == incident_id), None)
    if incident is None:
        raise HTTPException(status_code=404, detail="Incident not found")
    if not incident.get("has_clip"):
        incident["status"] = "RECORDING_FAILED"
        save_incidents()
        await sio.emit("recording_failed", {"incident_id": incident_id})
    return {"success": True}


# Static Files & Dashboard Routes
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir, exist_ok=True)

@app.delete("/api/incidents/{incident_id}/video")
async def delete_incident_video(incident_id: str):
    import re
    if not re.fullmatch(r"INC-[A-Za-z0-9-]+", incident_id):
        raise HTTPException(400, "Invalid incident ID")
    incident = next((item for item in incidents_db if item["id"] == incident_id), None)
    if incident is None:
        raise HTTPException(404, "Incident not found")
    if incident.get("status") == "RECORDING_EVIDENCE":
        raise HTTPException(409, "Wait until recording finishes before deleting")
    for prefix in ("clip_", "raw_"):
        for extension in (".mp4", ".webm", ".ogv", ".tmp.mp4"):
            path = os.path.join(CLIPS_DIR, prefix + incident_id + extension)
            if os.path.isfile(path):
                os.remove(path)
    incident.update(has_clip=False, clip_url=None, clip_bytes=0,
                    status="EVIDENCE_DELETED", video_deleted=True)
    save_incidents()
    return {"success": True}


app.mount("/static/clips", StaticFiles(directory=CLIPS_DIR), name="clips")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/ambulance")
@app.get("/ambulance.html")
async def ambulance_dashboard():
    return await root()


@app.get("/police")
@app.get("/police.html")
async def police_dashboard():
    return await root()


@app.get("/vision")
@app.get("/vision.html")
@app.get("/admin")
@app.get("/admin.html")
async def vision_dashboard():
    return await root()


@app.get("/")
async def root():
    # The bundled one-file frontend is normally named index.html. Also
    # support Dashboard.jsx when an operator has renamed the HTML bundle;
    # it contains standalone HTML, so explicitly return it as text/html.
    for frontend_name in ("index.html", "Dashboard.jsx"):
        frontend_file = os.path.join(static_dir, frontend_name)
        if os.path.exists(frontend_file):
            return FileResponse(frontend_file, media_type="text/html")
    return HTMLResponse("<h1>AcciDEN AI Server running</h1><p>Unified frontend loading...</p>")


# Combined ASGI application wrapping FastAPI and Socket.IO
combined_app = socketio.ASGIApp(sio, other_asgi_app=app)


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*65)
    print("  🚀 AcciDEN AI - Collision & Traffic Intelligence Hub")
    print("  🌐 Unified Command Page: http://127.0.0.1:8000")
    print("  👁️  Vision Dashboard:    http://127.0.0.1:8000/vision")
    print("  🚑 Ambulance Dashboard: http://127.0.0.1:8000/ambulance")
    print("  ⚡ Real-Time Socket.IO: ws://127.0.0.1:8000/socket.io")
    print("  📊 REST API Docs:       http://127.0.0.1:8000/docs")
    print("="*65 + "\n")
    from launch_app import main
    raise SystemExit(main())
