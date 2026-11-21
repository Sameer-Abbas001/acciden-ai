import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from backend.detection import summarize_sequence


def accident(bbox=(10, 10, 80, 80), score=80):
    return {'class_name': 'accident', 'is_accident': True,
            'confidence_pct': score, 'bbox': list(bbox)}


class ConfirmationPolicy(unittest.TestCase):
    def test_single_frame_never_confirms_even_high_confidence(self):
        _, summary = summarize_sequence([[accident(score=99)]])
        self.assertFalse(summary['accident_detected'])
        self.assertTrue(summary['candidate_detected'])

    def test_isolated_latest_prediction_stays_candidate(self):
        boxes, summary = summarize_sequence([[], [], [accident()]])
        self.assertFalse(summary['accident_detected'])
        self.assertFalse(boxes[0]['confirmed'])

    def test_matching_latest_region_confirms_two_of_three(self):
        _, summary = summarize_sequence([[accident()], [], [accident((12, 11, 82, 81))]])
        self.assertTrue(summary['accident_detected'])

    def test_predictions_in_unrelated_regions_do_not_confirm(self):
        _, summary = summarize_sequence([[accident((100, 100, 150, 150))], [], [accident()]])
        self.assertFalse(summary['accident_detected'])

    def test_disappeared_candidate_does_not_trigger_delayed_alert(self):
        _, summary = summarize_sequence([[accident()], [accident()], []])
        self.assertFalse(summary['accident_detected'])

    def test_endpoint_uses_training_resolution_and_preview_does_not_save(self):
        import cv2
        import numpy as np
        import base64
        from fastapi.testclient import TestClient
        from api.index import app
        _, image = cv2.imencode('.jpg', np.zeros((64, 64, 3), dtype=np.uint8))
        encoded = base64.b64encode(image.tobytes()).decode()
        box = SimpleNamespace(cls=[0], conf=[.8], xyxy=[np.array([10, 10, 50, 50])])
        detector = SimpleNamespace(names={0:'accident'})
        from unittest.mock import Mock
        detector.predict = Mock(return_value=[SimpleNamespace(boxes=[box]) for _ in range(3)])
        with patch('api.index.get_detection_model', return_value=detector), patch('backend.storage.create') as create:
            response = TestClient(app).post('/api/detect', data={
                'frames_json': json.dumps([encoded]*3), 'preview_only':'true'})
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json()['summary']['accident_detected'])
            self.assertIsNone(response.json()['incident'])
            self.assertEqual(detector.predict.call_args.kwargs['imgsz'], 512)
            create.assert_not_called()

    def test_single_frame_endpoint_does_not_create_incident(self):
        import cv2
        import numpy as np
        import base64
        from fastapi.testclient import TestClient
        from api.index import app
        from unittest.mock import Mock
        _, image = cv2.imencode('.jpg', np.zeros((64, 64, 3), dtype=np.uint8))
        box = SimpleNamespace(cls=[0], conf=[.99], xyxy=[np.array([10, 10, 50, 50])])
        detector = SimpleNamespace(names={0:'accident'}, predict=Mock(return_value=[SimpleNamespace(boxes=[box])]))
        with patch('api.index.get_detection_model', return_value=detector), patch('backend.storage.create') as create:
            response = TestClient(app).post('/api/detect', data={
                'base64_image':base64.b64encode(image.tobytes()).decode()})
            self.assertFalse(response.json()['summary']['accident_detected'])
            create.assert_not_called()
