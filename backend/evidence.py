"""Finalize browser recordings before publishing evidence to response teams."""
import os
import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg
from fastapi import HTTPException


def finalize_recording(contents, suffix):
    # MediaRecorder WebM often has no trustworthy duration/frame-count metadata.
    # Decode and remux through FFmpeg rather than estimating length from metadata.
    with tempfile.TemporaryDirectory(prefix='acciden-clip-') as directory:
        source = Path(directory) / ('recording' + suffix)
        target = Path(directory) / 'evidence.mp4'
        source.write_bytes(contents)
        command = [imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-hide_banner',
                   '-loglevel', 'error', '-xerror', '-i', str(source),
                   '-map', '0:v:0', '-an', '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2,fps=30',
                   '-c:v', 'libx264', '-threads', '1', '-preset', 'veryfast',
                   '-crf', '28', '-pix_fmt', 'yuv420p', '-t', '8',
                   '-movflags', '+faststart', '-progress', 'pipe:1', '-nostats', str(target)]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            raise HTTPException(422, 'Video processing timed out. Try a lower-resolution source.')
        if result.returncode != 0 or not target.exists() or target.stat().st_size < 512:
            raise HTTPException(422, 'The recording could not be decoded. Keep the camera connected and retry.')
        timestamps = []
        for line in result.stdout.splitlines():
            if line.startswith('out_time_us='):
                try:
                    timestamps.append(int(line.split('=', 1)[1]) / 1_000_000)
                except ValueError:
                    pass
        duration = max(timestamps, default=0)
        if duration < 7.75:
            raise HTTPException(422, f'The recording contains only {duration:.2f} seconds. Keep Admin visible and the source playing for the full capture.')
        if target.stat().st_size > 3_000_000:
            raise HTTPException(413, 'Processed video exceeds 3 MB. Try a lower-resolution source.')
        return target.read_bytes(), round(duration, 3)
