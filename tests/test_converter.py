
import pytest
import app

def test_convert_legacy_structure():
    # Legacy: {'a': {'b': {}, 'c': {}}} -> {'a': {'b': ['b'], 'c': ['c']}} ?
    # No, b is leaf, c is leaf.
    # Parent 'a' has children b (leaf), c (leaf).
    # All children are leaves -> 'a': ['b', 'c']

    data = {
        'root': {
            'leaf1': {},
            'leaf2': {}
        }
    }
    result = app.convert_to_wildcard_format(data)
    assert 'root' in result
    # We expect ['leaf1', 'leaf2'] (order might vary, but likely preserved)
    assert set(result['root']) == {'leaf1', 'leaf2'}

def test_convert_legacy_mixed():
    # root -> leaf1, sub -> leaf2
    data = {
        'root': {
            'leaf1': {},
            'sub': {
                'leaf2': {}
            }
        }
    }
    # Old Expected:
    # root:
    #   leaf1: [leaf1]
    #   sub: [leaf2]
    # New Expected: Mixed List
    # root:
    #   - leaf1
    #   - sub:
    #       - leaf2

    result = app.convert_to_wildcard_format(data)
    root = result['root']
    assert isinstance(root, list)
    assert 'leaf1' in root
    # Find the dict for sub
    sub_node = next((item for item in root if isinstance(item, dict) and 'sub' in item), None)
    assert sub_node is not None
    assert sub_node['sub'] == ['leaf2']

def test_convert_tree_structure():
    # Tree: {'root': {'leaf1': 'leaf1', 'leaf2': 'leaf2'}}
    data = {
        'root': {
            'leaf1': 'leaf1',
            'leaf2': 'leaf2'
        }
    }
    result = app.convert_to_wildcard_format(data)
    assert set(result['root']) == {'leaf1', 'leaf2'}

def test_convert_tree_mixed():
    # root -> leaf1, sub -> leaf2
    data = {
        'root': {
            'leaf1': 'leaf1',
            'sub': {
                'leaf2': 'leaf2'
            }
        }
    }
    result = app.convert_to_wildcard_format(data)
    root = result['root']
    assert isinstance(root, list)
    assert 'leaf1' in root
    sub_node = next((item for item in root if isinstance(item, dict) and 'sub' in item), None)
    assert sub_node is not None
    assert sub_node['sub'] == ['leaf2']

def test_convert_coco_style():
    # {'super': ['c1', 'c2']} -> Preserved
    data = {'super': ['c1', 'c2']}
    result = app.convert_to_wildcard_format(data)
    assert result == data

def test_deep_recursion():
    data = {'a': {'b': {'c': {'d': {}}}}}
    # a -> b -> c -> [d]
    result = app.convert_to_wildcard_format(data)
    assert result['a']['b']['c'] == ['d']
