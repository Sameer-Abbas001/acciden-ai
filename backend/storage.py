"""Private Vercel Blob storage with a lease protecting the incident manifest."""
import json
import os
import time
from contextlib import contextmanager

import httpx
from fastapi import HTTPException
from vercel.blob import sync as blob

MANIFEST = 'acciden/incidents.json'
LOCK = 'acciden/manifest.lock'


def read_json(path, default=None):
    try:
        metadata = blob.head_blob(path)
    except blob.BlobNotFoundError:
        return default
    # Manifest must bypass CDN caches so updates in other instances are visible.
    token = os.getenv('BLOB_READ_WRITE_TOKEN', '')
    if not token:
        raise HTTPException(503, 'Connect a private Blob store and set BLOB_READ_WRITE_TOKEN')
    response = httpx.get(metadata.url, params={'cache': '0'},
                        headers={'Authorization': 'Bearer ' + token, 'Cache-Control': 'no-cache'}, timeout=30)
    response.raise_for_status()
    return response.json()


@contextmanager
def locked():
    lease = None
    for attempt in range(8):
        try:
            lease = blob.put_blob(LOCK, json.dumps({'expires': time.time() + 330}).encode(),
                                  access='private', content_type='application/json', cache_control_max_age=0)
            break
        except blob.BlobError:
            # A timed-out invocation may leave a lease. Conditional deletion
            # ensures an expired lease cannot remove a newer owner's lock.
            try:
                metadata = blob.head_blob(LOCK)
                saved = read_json(LOCK)
                if saved and saved['expires'] < time.time():
                    blob.delete_blob(LOCK, if_match=metadata.etag)
            except blob.BlobNotFoundError:
                pass
            if attempt == 7:
                raise HTTPException(503, 'Storage busy. Please retry in a moment.')
            time.sleep(.2)
    try:
        yield
    finally:
        if lease:
            blob.delete_blob(LOCK, if_match=lease.etag)


def load():
    data = read_json(MANIFEST, {'rows': {}, 'last_detection': 0})
    for row in data['rows'].values():
        item = row['payload']
        # A stale manifest read used to lose deletion flags during the final
        # cleanup write. That write cleared the object path and byte count.
        if item.get('has_clip') and not item.get('object_path') and item.get('clip_bytes') == 0:
            item.update(has_clip=False, video_deleted=True, clip_url=None, status='EVIDENCE_DELETED')
    return data


def save(data):
    blob.put_blob(MANIFEST, json.dumps(data).encode(), access='private',
                  content_type='application/json', allow_overwrite=True, cache_control_max_age=0)


def rows():
    return list(load()['rows'].values())


def get(incident_id):
    row = load()['rows'].get(incident_id)
    if row is None:
        raise HTTPException(404, 'Incident not found')
    return row


def update(row, changes):
    with locked():
        data = load()
        current = data['rows'].get(row['id'])
        if current is None or current['version'] != row['version']:
            raise HTTPException(409, 'Incident changed in another tab. Refresh and retry.')
        payload = {**current['payload'], **changes}
        data['rows'][row['id']] = {'id': row['id'], 'version': row['version'] + 1, 'payload': payload}
        save(data)
        return payload


def create(incident_id, payload):
    with locked():
        data = load()
        if time.time() - data['last_detection'] < 12:
            return False
        data['rows'][incident_id] = {'id': incident_id, 'version': 0, 'payload': payload}
        data['last_detection'] = time.time()
        save(data)
        return True


def put_video(path, contents, content_type):
    blob.put_blob('acciden/videos/' + path, contents, access='private', content_type=content_type)


def delete_video(path):
    blob.delete_blob('acciden/videos/' + path)


def video(path):
    result = blob.get_blob('acciden/videos/' + path, access='private')
    return result.body, result.metadata.content_type
