"""Batch folders for the start frame -> end frame -> video pipeline.

Stdlib only (no torch, numpy or gradio), so it can be unit-tested without a GPU.

Every run gets its own folder, which lives on Google Drive when outputs/ is symlinked there:

    <output_root>/<batch_id>/
        manifest.json     prompts, settings and per-item status
        start/            the start frames, one per video
        end/              uploaded or generated end frames
        review/           start|end side-by-side thumbnails for the review gallery
        videos/           finished videos

The manifest is rewritten atomically after every item, so a Colab session that dies mid-batch
can be resumed: finished items are skipped.
"""

import difflib
import json
import os
import re
import shutil
import tempfile
import time
import zipfile

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff')
MANIFEST_NAME = 'manifest.json'
MANIFEST_VERSION = 1


def natural_sort_key(path):
    name = os.path.basename(path).lower()
    return [int(part) if part.isdigit() else part for part in re.split(r'(\d+)', name)]


def sanitize_name(name, fallback='image'):
    stem = os.path.splitext(os.path.basename(name))[0]
    stem = re.sub(r'[^A-Za-z0-9._-]+', '_', stem).strip('._')
    return (stem or fallback)[:64]


def is_image_path(path):
    return path.lower().endswith(IMAGE_EXTENSIONS)


def extract_zip_images(zip_path, dest_dir):
    """Extract every image in the archive into dest_dir, flattened so archive paths cannot escape it."""
    extracted = []
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = info.filename.replace('\\', '/')
            if name.startswith('__MACOSX/'):
                continue
            base = os.path.basename(name)
            if not base or base.startswith('.') or not is_image_path(base):
                continue

            stem, ext = os.path.splitext(base)
            target = os.path.join(dest_dir, base)
            counter = 1
            while os.path.exists(target):
                target = os.path.join(dest_dir, f'{stem}_{counter}{ext}')
                counter += 1

            with zf.open(info) as src, open(target, 'wb') as dst:
                shutil.copyfileobj(src, dst)
            extracted.append(target)

    extracted.sort(key=natural_sort_key)
    return extracted


def expand_image_inputs(paths, scratch_dir):
    """Turns a mix of image files and ZIP archives into a sorted list of image paths."""
    images = []
    for path in paths or []:
        if not path:
            continue
        if path.lower().endswith('.zip'):
            zip_dir = tempfile.mkdtemp(prefix='zip_', dir=scratch_dir)
            images.extend(extract_zip_images(path, zip_dir))
        elif is_image_path(path):
            images.append(path)
        else:
            print(f'Skipping {os.path.basename(path)}: not an image or ZIP file')
    images.sort(key=natural_sort_key)
    return images


# Words that only say which end of a pair an image is, so they are ignored when matching names.
PAIR_MARKERS = frozenset((
    'start', 'starting', 'begin', 'beginning', 'first', 'src', 'source', 'input',
    'end', 'ending', 'finish', 'final', 'last', 'dst', 'dest', 'destination', 'target', 'output',
))


def match_key(name):
    """Normalized name used to pair start/end images: drops pair markers and zero padding."""
    stem = os.path.splitext(os.path.basename(name))[0].lower()
    tokens = re.findall(r'[a-z]+|\d+', stem)
    normalized = [str(int(t)) if t.isdigit() else t for t in tokens]
    kept = [t for t in normalized if t not in PAIR_MARKERS]
    return '_'.join(kept or normalized)


def pair_images_by_name(start_paths, end_paths):
    """Pairs every start image with the closest-named end image, one end image per start.

    Pairs are assigned best score first, so confident name matches win before the leftovers.
    Names with nothing in common all score alike and, because the sort is stable, fall back to
    the order the images appear in their archives.
    """
    pairs = {path: None for path in start_paths}
    if not end_paths:
        return pairs

    keys = {path: match_key(path) for path in list(start_paths) + list(end_paths)}
    scored = []
    for start_path in start_paths:
        for end_path in end_paths:
            ratio = difflib.SequenceMatcher(None, keys[start_path], keys[end_path]).ratio()
            scored.append((ratio, start_path, end_path))
    scored.sort(key=lambda item: -item[0])

    matched_ends = set()
    for _, start_path, end_path in scored:
        if pairs[start_path] is not None or end_path in matched_ends:
            continue
        pairs[start_path] = end_path
        matched_ends.add(end_path)

    return pairs


def unique_names(paths):
    """Maps each path to a filesystem-safe item name, suffixing _2, _3, ... on collisions."""
    names = {}
    used = set()
    for path in paths:
        base = sanitize_name(path)
        name = base
        counter = 2
        while name.lower() in used:
            name = f'{base}_{counter}'
            counter += 1
        used.add(name.lower())
        names[path] = name
    return names


def new_batch_id():
    return time.strftime('%y%m%d_%H%M%S')


def _write_json_atomic(path, data):
    tmp_path = f'{path}.tmp'
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, path)


class Batch:
    """One batch folder and its manifest. Item paths in the manifest are relative to the folder."""

    def __init__(self, root):
        self.root = os.path.abspath(root)
        with open(os.path.join(self.root, MANIFEST_NAME), encoding='utf-8') as f:
            self.manifest = json.load(f)

    @property
    def batch_id(self):
        return self.manifest['batch_id']

    @property
    def names(self):
        return list(self.manifest['order'])

    def item(self, name):
        return self.manifest['items'][name]

    def _abs(self, rel_path):
        return os.path.normpath(os.path.join(self.root, rel_path)) if rel_path else None

    def start_path(self, name):
        return self._abs(self.item(name)['start'])

    def end_path(self, name):
        """The item's end frame, or None if it has none (yet)."""
        path = self._abs(self.item(name).get('end'))
        return path if path and os.path.exists(path) else None

    def generated_end_path(self, name):
        return os.path.join(self.root, 'end', f'{name}.png')

    def review_path(self, name):
        return os.path.join(self.root, 'review', f'{name}.jpg')

    def video_path(self, name):
        return os.path.join(self.root, 'videos', f'{name}.mp4')

    def video_done(self, name):
        # The manifest only says 'done' after the file is fully written, so a file left half-copied
        # by a dead session is redone.
        return self.item(name).get('video_status') == 'done' and os.path.exists(self.video_path(name))

    def pending_end_frames(self):
        return [name for name in self.names if self.end_path(name) is None]

    def pending_videos(self):
        return [name for name in self.names if not self.video_done(name)]

    def generated_names(self):
        return [name for name in self.names if self.item(name).get('end_source') == 'generated']

    def update(self, **fields):
        self.manifest.update(fields)
        self.save()

    def update_item(self, name, **fields):
        self.item(name).update(fields)
        self.save()

    def record_end(self, name, path, source, seed=None, prompt=None, seconds=None):
        """Points the item at a new end frame and deletes the one it replaces."""
        item = self.item(name)
        old_path = self._abs(item.get('end'))
        new_rel = os.path.relpath(path, self.root).replace('\\', '/')
        if old_path and os.path.abspath(old_path) != os.path.abspath(path) and os.path.exists(old_path):
            os.remove(old_path)
        item.update(end=new_rel, end_source=source, end_seed=seed, end_prompt=prompt)
        if seconds is not None:
            item.setdefault('timings', {})['end_frame'] = round(seconds, 2)
        # A new end frame makes any existing video stale.
        if item.get('video_status') == 'done':
            item['video_status'] = 'stale'
        self.save()

    def record_video(self, name, status, seconds=None, end_gap=None):
        item = self.item(name)
        item['video_status'] = status
        if status == 'done':
            item['video'] = f'videos/{name}.mp4'
        if seconds is not None:
            item.setdefault('timings', {})['video'] = round(seconds, 2)
        item['end_gap'] = None if end_gap is None else round(float(end_gap), 2)
        self.save()

    def save(self):
        _write_json_atomic(os.path.join(self.root, MANIFEST_NAME), self.manifest)

    def summary(self):
        total = len(self.names)
        ends = total - len(self.pending_end_frames())
        videos = total - len(self.pending_videos())
        return f'{total} item(s), {ends} end frame(s), {videos} video(s) done'


def create_batch(output_root, start_paths, end_paths=(), fallback_end_path=None, settings=None, batch_id=None):
    """Creates a batch folder from start images and optional end images.

    End images are paired with start images by closest file name. fallback_end_path, if given, is
    used as the end frame of every start image that found no partner.
    """
    if not start_paths:
        raise ValueError('No start images given.')

    batch_id = batch_id or new_batch_id()
    root = os.path.join(output_root, batch_id)
    counter = 2
    while os.path.exists(root):
        root = os.path.join(output_root, f'{batch_id}_{counter}')
        counter += 1
    batch_id = os.path.basename(root)
    for sub in ('start', 'end', 'review', 'videos'):
        os.makedirs(os.path.join(root, sub))

    names = unique_names(start_paths)
    pairs = pair_images_by_name(list(start_paths), list(end_paths))
    items = {}
    for path in start_paths:
        name = names[path]
        ext = os.path.splitext(path)[1].lower() or '.png'
        start_rel = f'start/{name}{ext}'
        shutil.copyfile(path, os.path.join(root, start_rel))

        item = dict(start=start_rel, end=None, end_source=None, end_seed=None, end_prompt=None,
                    video=None, video_status='pending', end_gap=None, timings={})
        end_src = pairs[path] or fallback_end_path
        if end_src:
            end_rel = f'end/{name}{os.path.splitext(end_src)[1].lower() or ".png"}'
            shutil.copyfile(end_src, os.path.join(root, end_rel))
            item.update(end=end_rel, end_source='uploaded')
        items[name] = item

    manifest = dict(
        version=MANIFEST_VERSION,
        batch_id=batch_id,
        created=time.strftime('%Y-%m-%d %H:%M:%S'),
        edit_prompt=None,
        motion_prompt=None,
        settings=settings or {},
        order=[names[path] for path in start_paths],
        items=items,
    )
    _write_json_atomic(os.path.join(root, MANIFEST_NAME), manifest)
    return Batch(root)


def list_batches(output_root):
    """Every batch folder under output_root, newest first, as (label, path) pairs for a dropdown."""
    if not os.path.isdir(output_root):
        return []
    found = []
    for entry in os.listdir(output_root):
        root = os.path.join(output_root, entry)
        if not os.path.isfile(os.path.join(root, MANIFEST_NAME)):
            continue
        try:
            batch = Batch(root)
        except (OSError, ValueError, KeyError):
            continue
        found.append((batch.manifest.get('created', ''), f'{batch.batch_id} ({batch.summary()})', root))
    found.sort(reverse=True)
    return [(label, root) for _, label, root in found]


def list_resumable(output_root):
    """Batches that still have videos left to make."""
    return [(label, root) for label, root in list_batches(output_root) if Batch(root).pending_videos()]
