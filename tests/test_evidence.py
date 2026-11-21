import subprocess
import tempfile
import unittest
from pathlib import Path
import imageio_ffmpeg
from fastapi import HTTPException
from backend.evidence import finalize_recording


class BrowserEvidence(unittest.TestCase):
    def recording(self, seconds):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'browser.webm'
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error',
                            '-f', 'lavfi', '-i', 'color=c=black:s=64x64:r=30',
                            '-t', str(seconds), '-c:v', 'libvpx', '-live', '1', str(output)],
                           check=True, capture_output=True)
            return output.read_bytes()

    def test_browser_webm_without_final_duration_becomes_playable_mp4(self):
        contents, duration = finalize_recording(self.recording(8.25), '.webm')
        self.assertIn(b'ftyp', contents[:32])
        self.assertGreaterEqual(duration, 7.75)
        self.assertLessEqual(duration, 8.1)

    def test_incomplete_recording_is_not_padded_into_evidence(self):
        with self.assertRaises(HTTPException) as error:
            finalize_recording(self.recording(2), '.webm')
        self.assertEqual(error.exception.status_code, 422)
        self.assertIn('only', error.exception.detail)

    def test_corrupt_recording_is_rejected(self):
        with self.assertRaises(HTTPException):
            finalize_recording(b'not a playable video' * 100, '.webm')
