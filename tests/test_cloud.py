import os
import unittest
import copy
from contextlib import nullcontext
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from api.index import app, update
from fastapi import HTTPException


class CloudAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.headers = {}

    def test_dashboards_work_without_password_configuration(self):
        with patch.dict(os.environ, {'APP_PASSWORD': ''}):
            self.assertEqual(self.client.get('/api/status').status_code, 200)
            with patch('backend.storage.rows', return_value=[]):
                self.assertEqual(self.client.get('/api/response/ambulance/incidents').status_code, 200)
                self.assertEqual(self.client.get('/api/response/police/incidents').status_code, 200)

    def test_update_rejects_concurrent_modifications(self):
        with patch('backend.storage.locked'), patch('backend.storage.load', return_value={'rows': {'INC-test': {'version': 3}}, 'last_detection': 0}):
            with self.assertRaises(HTTPException) as error:
                update({'id': 'INC-test', 'version': 2, 'payload': {}}, {'has_clip': False})
            self.assertEqual(error.exception.status_code, 409)

    def test_storage_failure_returns_json_instead_of_plain_server_error(self):
        with patch('backend.storage.rows', side_effect=RuntimeError('test storage failure')):
            response = self.client.get('/api/response/ambulance/incidents')
            self.assertEqual(response.status_code, 500)
            self.assertIn('runtime logs', response.json()['detail'])
            self.assertNotIn('test storage failure', response.text)

    def test_delete_removes_object_and_preserves_incident(self):
        row = {'id': 'INC-test', 'version': 0, 'payload': {'object_path': 'INC-test.webm', 'has_clip': True}}
        with patch('api.index.record', return_value=row), patch('api.index.update') as write, patch('backend.storage.delete_video') as remote:
            response = self.client.delete('/api/incidents/INC-test/video', headers=self.headers)
            self.assertEqual(response.status_code, 200)
            self.assertTrue(write.call_args_list[0].args[1]['video_deleted'])
            remote.assert_called_once_with('INC-test.webm')
            self.assertEqual(write.call_args_list[1].args[1]['clip_bytes'], 0)

    def test_failed_storage_delete_leaves_retry_reference(self):
        row = {'id': 'INC-test', 'version': 0, 'payload': {'object_path': 'INC-test.webm', 'has_clip': True}}
        with patch('api.index.record', return_value=row), patch('api.index.update') as write, patch('backend.storage.delete_video', side_effect=HTTPException(502, 'storage unavailable')):
            response = self.client.delete('/api/incidents/INC-test/video', headers=self.headers)
            self.assertEqual(response.status_code, 502)
            self.assertEqual(write.call_count, 1)
            self.assertNotIn('object_path', write.call_args.args[1])

    def test_manifest_requests_latest_private_blob(self):
        from backend import storage
        with patch.dict(os.environ, {'BLOB_READ_WRITE_TOKEN': 'test'}), \
                patch.object(storage.blob, 'head_blob', return_value=Mock(url='https://example.test/manifest')), \
                patch.object(storage.httpx, 'get') as fetch:
            fetch.return_value.json.return_value = {'rows': {}}
            self.assertEqual(storage.load(), {'rows': {}})
            self.assertEqual(fetch.call_args.kwargs['params'], {'cache': '0'})

    def test_missing_recording_status_preserves_incident_and_allows_late_evidence(self):
        pending = {'id': 'INC-late', 'status': 'RECORDING_EVIDENCE', 'timestamp': '2020-01-01 00:00:00'}
        with patch('backend.storage.rows', return_value=[{'payload': pending}]):
            for team in ('ambulance', 'police'):
                items = self.client.get(f'/api/response/{team}/incidents').json()['incidents']
                self.assertEqual(items[0]['status'], 'EVIDENCE_MISSING')
        self.assertEqual(pending['status'], 'RECORDING_EVIDENCE')

    def test_retry_after_upload_acknowledgement_is_lost_preserves_clip(self):
        pending = {'id': 'INC-retry', 'version': 0, 'payload': {'has_clip': False}}
        complete = {'id': 'INC-retry', 'version': 1, 'payload': {'has_clip': True}}
        def published_then_failed(row, changes):
            complete['payload'].update(changes)
            raise TimeoutError('lease release timed out after publication')

        files = {'video_file': ('capture.webm', b'example' * 100, 'video/webm')}
        with patch('api.index.record', side_effect=[pending, complete, complete]), \
                patch('api.index.finalize_recording', return_value=(b'mp4', 8.0)), \
                patch('backend.storage.put_video') as put, patch('backend.storage.delete_video') as delete, \
                patch('api.index.update', side_effect=published_then_failed):
            for attempt in range(2):
                result = self.client.post('/api/clips/upload', data={'incident_id': 'INC-retry'}, files=files)
                self.assertEqual(result.status_code, 200, result.text)
            put.assert_called_once()
            delete.assert_not_called()

    def test_upload_retry_cannot_restore_deleted_video(self):
        with patch('api.index.record', return_value={'payload': {'video_deleted': True}}), \
                patch('backend.storage.put_video') as put:
            result = self.client.post('/api/clips/upload', data={'incident_id': 'INC-deleted'},
                                      files={'video_file': ('capture.webm', b'example' * 100, 'video/webm')})
            self.assertEqual(result.status_code, 409)
            put.assert_not_called()

    def test_separate_clients_share_video_decisions_and_admin_deletion(self):
        # Two independent browser clients read the same persistent manifest.
        from backend import storage
        database = {'rows': {}, 'last_detection': 0}

        def load():
            return copy.deepcopy(database)

        def save(data):
            database.clear()
            database.update(copy.deepcopy(data))

        payload = {'id': 'INC-shared', 'timestamp': '2026-11-21 12:00:00',
                   'has_clip': True, 'object_path': 'shared.webm', 'responses': {},
                   'status': 'AWAITING_CONFIRMATION'}
        with patch('backend.storage.locked', side_effect=lambda: nullcontext()), \
                patch('backend.storage.load', side_effect=load), \
                patch('backend.storage.save', side_effect=save), \
                patch('backend.storage.video', return_value=(b'0123456789', 'video/webm')), \
                patch('backend.storage.delete_video') as remove:
            storage.create('INC-shared', payload)
            ambulance = TestClient(app)
            police = TestClient(app)
            items = ambulance.get('/api/response/ambulance/incidents', headers=self.headers).json()['incidents']
            video_url = items[0]['clip_url']
            self.assertEqual(police.get(video_url).content, b'0123456789')
            partial = police.get(video_url, headers={'Range': 'bytes=2-5'})
            self.assertEqual(partial.status_code, 206)
            self.assertEqual(partial.content, b'2345')
            self.assertEqual(ambulance.post('/api/response/ambulance/confirm', headers=self.headers,
                                           data={'incident_id': 'INC-shared'}).status_code, 200)
            self.assertEqual(police.post('/api/response/police/reject', headers=self.headers,
                                        data={'incident_id': 'INC-shared'}).status_code, 200)
            latest = self.client.get('/api/response/ambulance/incidents', headers=self.headers).json()['incidents'][0]
            self.assertEqual(latest['responses']['ambulance']['status'], 'DISPATCHED')
            self.assertEqual(latest['responses']['police']['status'], 'REJECTED')
            self.assertEqual(self.client.delete('/api/incidents/INC-shared/video', headers=self.headers).status_code, 200)
            remove.assert_called_once_with('shared.webm')
            self.assertEqual(police.get(video_url).status_code, 404)
            retained = ambulance.get('/api/response/ambulance/incidents', headers=self.headers).json()['incidents'][0]
            self.assertTrue(retained['video_deleted'])
            self.assertFalse(retained['has_clip'])


if __name__ == '__main__':
    unittest.main()

