import base64
import json
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

from api.index import app


def encoded_frame(brightness):
    image = np.full((64, 64, 3), brightness, dtype=np.uint8)
    _, encoded = cv2.imencode('.png', image)
    return base64.b64encode(encoded.tobytes()).decode()


def box(class_id, score):
    return SimpleNamespace(cls=[class_id], conf=[score], xyxy=[[10.2, 11.9, 50.5, 51.3]])


class FrameDetection(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.model = Mock(names={0: 'accident', 1: 'vehicle', 2: 'truck'})
        mocks = ExitStack()
        self.addCleanup(mocks.close)
        mocks.enter_context(patch('api.index.get_detection_model', return_value=self.model))
        self.create = mocks.enter_context(patch('backend.storage.create', return_value=True))

    def test_single_accident_uses_script_defaults_and_creates_incident(self):
        self.model.return_value = [SimpleNamespace(boxes=[box(0, .27), box(1, .88)])]
        response = self.client.post('/api/detect', data={'base64_image': encoded_frame(0)})
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertEqual(self.model.call_args.kwargs, {
            'conf': .25, 'imgsz': 320, 'verbose': False, 'device': 'cpu'})
        self.assertEqual(self.model.call_args.args[0].shape, (64, 64, 3))
        self.assertEqual(data['detections'][0]['bbox'], [10, 11, 50, 51])
        self.assertTrue(data['detections'][0]['confirmed'])
        self.assertFalse(data['detections'][1]['is_accident'])
        self.assertTrue(data['summary']['accident_detected'])
        self.assertEqual(data['summary']['vehicles_count'], 1)
        self.assertEqual(data['incident']['confidence'], 27)
        self.create.assert_called_once()

    def test_vehicle_only_and_empty_frames_do_not_create_incidents(self):
        for predictions in ([box(2, .98)], []):
            with self.subTest(predictions=predictions):
                self.model.return_value = [SimpleNamespace(boxes=predictions)]
                response = self.client.post('/api/detect', data={'base64_image': encoded_frame(0)})
                self.assertEqual(response.status_code, 200, response.text)
                data = response.json()
                self.assertFalse(data['summary']['accident_detected'])
                self.assertEqual(data['summary']['vehicles_count'], len(predictions))
                self.assertIsNone(data['incident'])
        self.create.assert_not_called()

    def test_legacy_batch_analyzes_displayed_frame_and_respects_preview(self):
        self.model.return_value = [SimpleNamespace(boxes=[box(0, .65)])]
        response = self.client.post('/api/detect', data={
            'frames_json': json.dumps([encoded_frame(v) for v in (0, 100, 220)]),
            'preview_only': 'true', 'imgsz': '512', 'conf': '.6'})
        self.assertEqual(response.status_code, 200, response.text)
        self.model.assert_called_once()
        np.testing.assert_array_equal(self.model.call_args.args[0], np.full((64, 64, 3), 220))
        self.assertEqual(self.model.call_args.kwargs['conf'], .6)
        self.assertEqual(self.model.call_args.kwargs['imgsz'], 512)
        data = response.json()
        self.assertTrue(data['summary']['accident_detected'])
        self.assertEqual(data['summary']['frames_analyzed'], 1)
        self.assertEqual(data['summary']['minimum_alert_confidence'], 60)
        self.assertIsNone(data['incident'])
        self.create.assert_not_called()

    def test_local_preview_keeps_incident_storage_and_cooldown_available(self):
        import server
        self.model.return_value = [SimpleNamespace(boxes=[box(0, .27)])]
        with patch.object(server, 'get_model', return_value=self.model), \
                patch.object(server, 'last_server_accident_time', 0), \
                patch.object(server, 'save_incidents') as save:
            response = TestClient(server.app).post('/api/detect', data={
                'base64_image': encoded_frame(0), 'return_annotated': 'false', 'preview_only': 'true'})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertTrue(response.json()['summary']['accident_detected'])
            self.assertIsNone(response.json()['incident'])
            self.assertEqual(self.model.call_args.kwargs, {'conf': .25, 'imgsz': 320, 'verbose': False})
            self.assertEqual(server.last_server_accident_time, 0)
            save.assert_not_called()

    def test_local_accepts_legacy_camera_multipart_and_retries_the_latest_frame(self):
        import server
        latest = encoded_frame(220)
        samples = [encoded_frame(0), encoded_frame(100), 'data:image/png;base64,' + latest]
        self.model.side_effect = [[SimpleNamespace(boxes=[])], [SimpleNamespace(boxes=[box(0, .39)])]]
        with patch.object(server, 'get_model', return_value=self.model):
            response = TestClient(server.app).post('/api/detect', files={
                'frames_json': (None, json.dumps(samples)), 'live_camera': (None, 'true'),
                'preview_only': (None, 'true'), 'return_annotated': (None, 'false')})
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertTrue(data['summary']['accident_detected'])
        self.assertIsNone(data['incident'])
        self.assertEqual(data['summary']['inference_size'], 640)
        self.assertEqual([call.kwargs['imgsz'] for call in self.model.call_args_list], [320, 640])
        for call in self.model.call_args_list:
            np.testing.assert_array_equal(call.args[0], np.full((64, 64, 3), 220))

    def test_camera_retry_respects_boxes_and_selected_resolution(self):
        for size, predictions, expected in [(320, [box(1, .9)], [320]), (512, [], [512]), (640, [], [640]), (320, [], [320, 640])]:
            with self.subTest(size=size, expected=expected):
                self.model.reset_mock()
                self.model.return_value = [SimpleNamespace(boxes=predictions)]
                response = self.client.post('/api/detect', data={
                    'base64_image': encoded_frame(0), 'live_camera': 'true', 'imgsz': size})
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual([call.kwargs['imgsz'] for call in self.model.call_args_list], expected)
                self.assertFalse(response.json()['summary']['accident_detected'])
        self.create.assert_not_called()

    def test_local_invalid_camera_frames_return_client_error_before_inference(self):
        import server
        with patch.object(server, 'get_model') as get_model:
            client = TestClient(server.app)
            for fields in ({}, {'base64_image': 'data:,'}, {'frames_json': '[]'}, {'frames_json': 'broken'}):
                with self.subTest(fields=fields):
                    self.assertEqual(client.post('/api/detect', data=fields).status_code, 400)
            get_model.assert_not_called()
