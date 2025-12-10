import unittest
from unittest.mock import MagicMock
import app

class TestFlattening(unittest.TestCase):

    def setUp(self):
        self.original_wn = app.wn
        app.wn = MagicMock()

    def tearDown(self):
        app.wn = self.original_wn

    def make_synset_mock(self, name_str, offset_int):
        m = MagicMock()
        lemma = MagicMock()
        lemma.name.return_value = name_str
        m.lemmas.return_value = [lemma]
        m.pos.return_value = 'n'
        m.offset.return_value = offset_int
        # Default hyponyms to empty
        m.hyponyms.return_value = []
        return m

    def test_generic_flattening(self):
        data = {
            'A': {
                'B': 'leaf1',
                'C': {
                    'D': 'leaf2'
                }
            }
        }
        res = app.flatten_hierarchy_post_process(data, 0, 10)
        self.assertEqual(res, data)

        res_d1 = app.flatten_hierarchy_post_process(data, 0, 1)
        self.assertIsInstance(res_d1['A'], list)
        self.assertIn('leaf1', res_d1['A'])

    def test_imagenet_recursive_flattening(self):
        root = self.make_synset_mock('root', 1)
        child = self.make_synset_mock('child', 2)
        grandchild = self.make_synset_mock('grandchild', 3)

        root.hyponyms.return_value = [child]
        child.hyponyms.return_value = [grandchild]

        root.closure = MagicMock(side_effect=lambda func: iter([child, grandchild]))
        child.closure = MagicMock(side_effect=lambda func: iter([grandchild]))

        # Max Depth 0
        res = app.build_hierarchy_tree_recursive(root, None, 0, 0, strict_filter=False)
        self.assertEqual(sorted(res), ['child', 'grandchild'])

        # Max Depth 1
        res = app.build_hierarchy_tree_recursive(root, None, 0, 1, strict_filter=False)
        self.assertEqual(res, {'child': ['grandchild']})

    def test_extract_all_leaves(self):
        data = {'a': ['1', '2'], 'b': {'c': '3'}}
        leaves = app.extract_all_leaves(data)
        self.assertEqual(sorted(leaves), ['1', '2', '3'])

if __name__ == '__main__':
    unittest.main()
