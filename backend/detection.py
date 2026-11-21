"""Separate model candidates from spatially consistent multi-frame alerts."""


def overlap(first, second):
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, right-left) * max(0, bottom-top)
    first_area = max(0, first[2]-first[0]) * max(0, first[3]-first[1])
    second_area = max(0, second[2]-second[0]) * max(0, second[3]-second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0


def summarize_sequence(sequence):
    """Require the latest accident region to match at least one prior sample.

    This rejects isolated/jumping predictions. It cannot establish a real
    collision or reject a stable false positive without better model training.
    """
    latest = [dict(box) for box in sequence[-1]] if sequence else []
    for candidate in latest:
        hits = 1 if candidate['is_accident'] else 0
        if candidate['is_accident']:
            for earlier in sequence[:-1]:
                if any(box['is_accident'] and overlap(candidate['bbox'], box['bbox']) >= .25 for box in earlier):
                    hits += 1
        candidate['confirmation_frames'] = hits
        candidate['confirmed'] = candidate['is_accident'] and len(sequence) >= 3 and hits >= 2
    confirmed = [box for box in latest if box['confirmed']]
    candidates = [box for box in latest if box['is_accident']]
    return latest, {
        'accident_detected': bool(confirmed),
        'candidate_detected': bool(candidates),
        'max_accident_confidence': max((box['confidence_pct'] for box in confirmed), default=0),
        'max_candidate_confidence': max((box['confidence_pct'] for box in candidates), default=0),
        'vehicles_count': sum(box.get('class_name') == 'vehicle' for box in latest),
        'frames_analyzed': len(sequence),
        'confirmation_rule': 'latest region matches at least 2 of 3 frames',
        'inference_size': 512,
    }
