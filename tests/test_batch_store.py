import json
import os
import shutil
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pipeline import batch_store
from pipeline.batch_store import Batch, create_batch, list_batches, list_resumable, pair_images_by_name


def write_file(path, data=b'img'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        f.write(data)
    return path


class BatchStoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.inputs = os.path.join(self.tmp, 'inputs')
        self.outputs = os.path.join(self.tmp, 'outputs')
        os.makedirs(self.outputs)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_pairing_ignores_markers_and_padding(self):
        starts = ['a/shot03_start.png', 'a/shot1_start.png']
        ends = ['b/shot1_end.png', 'b/shot3_end.png']
        pairs = pair_images_by_name(starts, ends)
        self.assertEqual(pairs['a/shot03_start.png'], 'b/shot3_end.png')
        self.assertEqual(pairs['a/shot1_start.png'], 'b/shot1_end.png')

    def test_expand_inputs_mixes_images_and_zips(self):
        loose = write_file(os.path.join(self.inputs, 'b.png'))
        zip_path = os.path.join(self.inputs, 'more.zip')
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr('nested/a.jpg', b'x')
            zf.writestr('__MACOSX/nested/._a.jpg', b'x')
            zf.writestr('notes.txt', b'x')
        images = batch_store.expand_image_inputs([loose, zip_path, None], self.tmp)
        self.assertEqual([os.path.basename(p) for p in images], ['a.jpg', 'b.png'])

    def test_create_batch_names_pairs_and_fallback(self):
        s1 = write_file(os.path.join(self.inputs, 'x', 'shot 1_start.png'))
        s2 = write_file(os.path.join(self.inputs, 'y', 'shot 1_start.png'))  # same name, other folder
        s3 = write_file(os.path.join(self.inputs, 'shot2.jpg'))
        e1 = write_file(os.path.join(self.inputs, 'ends', 'shot1_end.png'), b'end1')
        fallback = write_file(os.path.join(self.inputs, 'fallback.png'), b'fb')

        batch = create_batch(self.outputs, [s1, s2, s3], [e1], fallback_end_path=fallback, batch_id='b1')

        self.assertEqual(batch.names, ['shot_1_start', 'shot_1_start_2', 'shot2'])
        self.assertTrue(os.path.exists(batch.start_path('shot2')))
        self.assertTrue(batch.start_path('shot2').endswith('.jpg'))
        # e1 goes to exactly one of the two same-named starts; everything else gets the fallback.
        sources = [open(batch.end_path(n), 'rb').read() for n in batch.names]
        self.assertEqual(sorted(sources), [b'end1', b'fb', b'fb'])
        self.assertEqual(batch.pending_end_frames(), [])

    def test_create_batch_never_overwrites_existing_folder(self):
        s1 = write_file(os.path.join(self.inputs, 'a.png'))
        first = create_batch(self.outputs, [s1], batch_id='same')
        second = create_batch(self.outputs, [s1], batch_id='same')
        self.assertNotEqual(first.root, second.root)
        self.assertEqual(second.batch_id, 'same_2')

    def test_manifest_round_trip_and_resume(self):
        starts = [write_file(os.path.join(self.inputs, f'{n}.png')) for n in ('a', 'b', 'c')]
        batch = create_batch(self.outputs, starts, batch_id='b1')
        self.assertEqual(batch.pending_end_frames(), ['a', 'b', 'c'])

        end = write_file(batch.generated_end_path('a'))
        batch.record_end('a', end, 'generated', seed=7, prompt='p', seconds=1.234)
        write_file(batch.video_path('a'))
        batch.record_video('a', 'done', seconds=10, end_gap=3.14159)
        # 'b' has a video file but its status never reached done (e.g. the session died mid-copy).
        write_file(batch.video_path('b'))

        reloaded = Batch(batch.root)
        self.assertEqual(reloaded.pending_end_frames(), ['b', 'c'])
        self.assertEqual(reloaded.pending_videos(), ['b', 'c'])
        self.assertEqual(reloaded.item('a')['end_seed'], 7)
        self.assertEqual(reloaded.item('a')['end_gap'], 3.14)
        self.assertEqual(reloaded.item('a')['timings'], {'end_frame': 1.23, 'video': 10})
        self.assertEqual(reloaded.generated_names(), ['a'])

        with open(os.path.join(batch.root, 'manifest.json'), encoding='utf-8') as f:
            self.assertEqual(json.load(f)['items']['a']['end'], 'end/a.png')

    def test_new_end_frame_replaces_old_and_marks_video_stale(self):
        start = write_file(os.path.join(self.inputs, 'a.png'))
        uploaded = write_file(os.path.join(self.inputs, 'a_end.jpg'))
        batch = create_batch(self.outputs, [start], [uploaded], batch_id='b1')
        old_end = batch.end_path('a')
        write_file(batch.video_path('a'))
        batch.record_video('a', 'done')

        new_end = write_file(batch.generated_end_path('a'))
        batch.record_end('a', new_end, 'generated', seed=1)

        self.assertFalse(os.path.exists(old_end))
        self.assertEqual(batch.end_path('a'), new_end)
        self.assertEqual(batch.pending_videos(), ['a'])

    def test_list_batches_and_resumable(self):
        start = write_file(os.path.join(self.inputs, 'a.png'))
        done = create_batch(self.outputs, [start], batch_id='done')
        write_file(done.video_path('a'))
        done.record_video('a', 'done')
        create_batch(self.outputs, [start], batch_id='todo')
        os.makedirs(os.path.join(self.outputs, 'not_a_batch'))

        self.assertEqual(len(list_batches(self.outputs)), 2)
        resumable = list_resumable(self.outputs)
        self.assertEqual([os.path.basename(root) for _, root in resumable], ['todo'])
        self.assertIn('0 video(s) done', resumable[0][0])


if __name__ == '__main__':
    unittest.main()
