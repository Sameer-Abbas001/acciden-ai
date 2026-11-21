"""Regression checks using isolated state; never writes real incident records."""
import asyncio
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import server
from fastapi import HTTPException, UploadFile
from starlette.datastructures import Headers


class ResponseWorkflow(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.original = server.incidents_db, server.STATE_FILE, server.CLIPS_DIR
        server.STATE_FILE = os.path.join(self.temp.name, 'incidents.json')
        server.CLIPS_DIR = self.temp.name
        server.incidents_db = [{'id': 'INC-TEST', 'has_clip': True, 'clip_url': '/static/clips/test.mp4', 'status': 'AWAITING_CONFIRMATION'}]
        self.events = patch.object(server.sio, 'emit', new_callable=AsyncMock)
        self.emit = self.events.start()

    def tearDown(self):
        server.incidents_db, server.STATE_FILE, server.CLIPS_DIR = self.original
        self.events.stop()
        self.temp.cleanup()

    async def test_agencies_share_evidence_but_review_independently(self):
        await server.response_action('ambulance', 'confirm', 'INC-TEST', 'Medical team')
        await server.response_action('police', 'reject', 'INC-TEST', 'No police response needed')
        for agency in ('ambulance', 'police'):
            item = (await server.response_incidents(agency))['incidents'][0]
            self.assertEqual(item['responses']['ambulance']['status'], 'DISPATCHED')
            self.assertEqual(item['responses']['police']['status'], 'REJECTED')
            self.assertEqual(item['clip_url'], '/static/clips/test.mp4')
        await server.response_action('ambulance', 'resolve', 'INC-TEST', '')
        self.assertTrue(Path(server.STATE_FILE).exists())

    async def test_missing_evidence_and_duplicate_decisions_are_blocked(self):
        server.incidents_db[0]['has_clip'] = False
        with self.assertRaises(HTTPException) as result:
            await server.response_action('police', 'confirm', 'INC-TEST', '')
        self.assertEqual(result.exception.status_code, 409)
        server.incidents_db[0]['has_clip'] = True
        await server.response_action('police', 'confirm', 'INC-TEST', '')
        with self.assertRaises(HTTPException):
            await server.response_action('police', 'reject', 'INC-TEST', '')
        with self.assertRaises(HTTPException):
            await server.response_incidents('unknown')

    async def test_detection_returns_recording_id_to_requesting_browser(self):
        import cv2
        import numpy as np
        _, encoded = cv2.imencode('.jpg', np.zeros((64,64,3), dtype=np.uint8))
        summary = {'accident_detected': True, 'max_accident_confidence': 95.0, 'severity_level': 'CRITICAL', 'vehicles_count': 2}
        with patch.object(server, 'process_cv2_frame', return_value=([], summary)), patch.object(server, 'last_server_accident_time', 0):
            result = await server.detect_api(file=UploadFile(io.BytesIO(encoded.tobytes()), filename='test.jpg'), base64_image=None, conf=.5, iou=.45, imgsz=320, return_annotated=False, location='Test junction')
        self.assertEqual(result['incident']['location'], 'Test junction')
        self.assertFalse(result['incident']['has_clip'])
        self.assertEqual(result['incident']['id'], server.incidents_db[0]['id'])

    async def test_real_recording_transcodes_and_reaches_both_agencies(self):
        raw = Path(__file__).resolve().parents[1] / 'static/clips/raw_INC-216248.webm'
        if not raw.exists():
            self.skipTest('Original camera recording is not available')
        upload = UploadFile(io.BytesIO(raw.read_bytes()), filename='evidence.webm', headers=Headers({'content-type':'video/webm'}))
        result = await server.upload_camera_clip('INC-TEST', upload, 95.0, 'Test junction', 2)
        self.assertTrue(result['success'])
        for agency in ('ambulance','police'):
            incident = (await server.response_incidents(agency))['incidents'][0]
            self.assertTrue(incident['has_clip'])
            self.assertEqual(incident['clip_url'], result['clip_url'])
        self.assertTrue(any(call.args[0] == 'clip_ready' for call in self.emit.call_args_list))

    async def test_empty_upload_does_not_publish_evidence(self):
        server.incidents_db[0]['has_clip'] = False
        with self.assertRaises(HTTPException):
            await server.upload_camera_clip('INC-TEST', UploadFile(io.BytesIO(b''), filename='empty.webm'), 95.0, 'Test', 2)
        self.assertFalse(server.incidents_db[0]['has_clip'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
