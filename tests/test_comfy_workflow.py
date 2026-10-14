import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pipeline.comfy_runner import ComfyError, apply_inputs, find_node, load_workflow

WORKFLOW_DIR = os.path.join(os.path.dirname(__file__), '..', 'pipeline', 'workflows')


class ComfyWorkflowTest(unittest.TestCase):
    def setUp(self):
        self.workflow = load_workflow(os.path.join(WORKFLOW_DIR, 'qwen_edit_2511.api.json'))

    def test_bundled_workflow_is_tagged_and_linked(self):
        for node_id, node in self.workflow.items():
            for value in node['inputs'].values():
                if isinstance(value, list):
                    self.assertIn(value[0], self.workflow, f'node {node_id} links to missing node {value[0]}')

    def test_apply_inputs_fills_tagged_nodes_without_mutating(self):
        wf = apply_inputs(self.workflow, 'start.png', 'make it night', 123)
        self.assertEqual(wf[find_node(wf, 'IN_IMAGE')]['inputs']['image'], 'start.png')
        self.assertEqual(wf[find_node(wf, 'IN_PROMPT')]['inputs']['prompt'], 'make it night')
        self.assertEqual(wf[find_node(wf, 'IN_SEED')]['inputs']['seed'], 123)
        self.assertEqual(self.workflow[find_node(self.workflow, 'IN_IMAGE')]['inputs']['image'], 'placeholder.png')

    def test_noise_seed_input_is_supported(self):
        wf = {
            '1': {'class_type': 'LoadImage', 'inputs': {'image': ''}, '_meta': {'title': 'IN_IMAGE'}},
            '2': {'class_type': 'Enc', 'inputs': {'prompt': ''}, '_meta': {'title': 'IN_PROMPT'}},
            '3': {'class_type': 'RandomNoise', 'inputs': {'noise_seed': 0}, '_meta': {'title': 'IN_SEED'}},
        }
        self.assertEqual(apply_inputs(wf, 'a.png', 'p', 9)['3']['inputs']['noise_seed'], 9)

    def test_missing_or_duplicate_tag_is_a_clear_error(self):
        wf = dict(self.workflow)
        del wf[find_node(wf, 'IN_PROMPT')]
        with self.assertRaisesRegex(ComfyError, 'IN_PROMPT.*found no node'):
            apply_inputs(wf, 'a.png', 'p', 1)

        wf = dict(self.workflow)
        wf['99'] = {'class_type': 'LoadImage', 'inputs': {'image': ''}, '_meta': {'title': 'IN_IMAGE'}}
        with self.assertRaisesRegex(ComfyError, 'found 2 nodes'):
            find_node(wf, 'IN_IMAGE')


if __name__ == '__main__':
    unittest.main()
