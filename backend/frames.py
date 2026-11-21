"""Decode the frame formats sent by current and older dashboards."""
import base64
import binascii
import json

from fastapi import HTTPException


def decode_image(contents):
    import cv2
    import numpy as np

    if contents:
        try:
            frame = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        except cv2.error:
            frame = None
        if frame is not None:
            return frame
    raise HTTPException(400, 'Invalid image frame')


def dashboard_frame(base64_image, frames_json, max_chars=None):
    try:
        images = json.loads(frames_json) if frames_json else [base64_image]
    except (ValueError, TypeError):
        raise HTTPException(400, 'Invalid frame sequence')
    if not isinstance(images, list) or len(images) not in (1, 3):
        raise HTTPException(400, 'Send one frame or a three-frame sequence')
    if any(not isinstance(image, str) or not image for image in images):
        raise HTTPException(400, 'Invalid image frame')
    if max_chars is not None and sum(map(len, images)) > max_chars:
        raise HTTPException(413, 'Frame sequence too large; use a lower source resolution')

    shape = None
    for image in images:
        try:
            contents = base64.b64decode(image.split(',', 1)[-1], validate=True)
        except (binascii.Error, ValueError):
            raise HTTPException(400, 'Invalid image frame')
        frame = decode_image(contents)
        if shape is not None and frame.shape != shape:
            raise HTTPException(400, 'All samples must come from the same video source')
        shape = frame.shape
    # Legacy dashboards display the boxes on the last image in the sequence.
    return frame
