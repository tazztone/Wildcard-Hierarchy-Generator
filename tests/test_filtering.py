import pytest
from unittest.mock import MagicMock, patch
import app

@pytest.fixture
def mock_wn_filtering():
    original_wn = app.wn
    mock_wn = MagicMock()
    app.wn = mock_wn
    yield mock_wn
    app.wn = original_wn

def test_get_primary_synset(mock_wn_filtering):
    # Mock wn.synsets
    s1 = MagicMock()
    s1.name.return_value = 'dog.n.01'
    s2 = MagicMock()
    s2.name.return_value = 'dog.n.02'

    mock_wn_filtering.synsets.return_value = [s1, s2]

    result = app.get_primary_synset('dog')
    assert result == s1
    mock_wn_filtering.synsets.assert_called_with('dog')

def test_get_primary_synset_none(mock_wn_filtering):
    mock_wn_filtering.synsets.return_value = []
    result = app.get_primary_synset('unknown')
    assert result is None

def test_strict_filtering_prunes_secondary(mock_wn_filtering):
    # Scenario: "old_man"
    # Current node: old_man.n.03 (slang)
    # Primary for "old_man": old_man.n.01

    current_synset = MagicMock()
    lemma = MagicMock()
    lemma.name.return_value = 'old_man'
    current_synset.lemmas.return_value = [lemma]
    current_synset.name.return_value = 'old_man.n.03'

    primary_synset = MagicMock()
    primary_synset.name.return_value = 'old_man.n.01'

    # Mock wn.synsets('old_man') to return [primary, current, ...]
    mock_wn_filtering.synsets.return_value = [primary_synset, current_synset]

    # Call recursive build with strict_filter=True
    # We expect it to return None because current != primary
    result = app.build_hierarchy_tree_recursive(
        current_synset, valid_wnids=None, depth=0, max_depth=3,
        strict_filter=True, blacklist=False
    )

    assert result is None

def test_strict_filtering_keeps_primary(mock_wn_filtering):
    # Scenario: "old_man"
    # Current node: old_man.n.01
    # Primary: old_man.n.01

    current_synset = MagicMock()
    lemma = MagicMock()
    lemma.name.return_value = 'old_man'
    current_synset.lemmas.return_value = [lemma]
    current_synset.name.return_value = 'old_man.n.01'

    # Mock wn.synsets('old_man') -> [current]
    mock_wn_filtering.synsets.return_value = [current_synset]

    # Mock children/closure to be empty leaf
    current_synset.hyponyms.return_value = []
    current_synset.closure.side_effect = lambda func: iter([])
    current_synset.pos.return_value = 'n'
    current_synset.offset.return_value = 12345

    result = app.build_hierarchy_tree_recursive(
        current_synset, valid_wnids=None, depth=0, max_depth=3,
        strict_filter=True, blacklist=False
    )

    assert result == 'old man'

def test_blacklist_prunes_category(mock_wn_filtering):
    # Scenario: "communication" node
    current_synset = MagicMock()
    lemma = MagicMock()
    lemma.name.return_value = 'communication' # displayed name
    current_synset.lemmas.return_value = [lemma]
    current_synset.name.return_value = 'communication.n.01' # Synset name

    # strict_filter=False to isolate blacklist test
    result = app.build_hierarchy_tree_recursive(
        current_synset, valid_wnids=None, depth=0, max_depth=3,
        strict_filter=False, blacklist=True
    )

    assert result is None

def test_hypernym_depth_limit(mock_wn_filtering):
    # Bottom-Up logic test (generate_imagenet_wnid_hierarchy)

    # Mock input WNID
    wnid = 'n12345'

    # Mock synset
    synset = MagicMock()
    mock_wn_filtering.synset_from_pos_and_offset.return_value = synset

    # Mock hypernym path: A -> B -> C -> D (Leaf)
    # Names: a, b, c, d
    node_a = MagicMock(); node_a.name.return_value = 'a.n.01'
    node_b = MagicMock(); node_b.name.return_value = 'b.n.01'
    node_c = MagicMock(); node_c.name.return_value = 'c.n.01'
    node_d = MagicMock(); node_d.name.return_value = 'd.n.01'

    synset.hypernym_paths.return_value = [[node_a, node_b, node_c, node_d]]

    # Test with hypernym_depth = 2
    # Should use only last 2: C -> D
    # Resulting tree should have 'c' as root, containing 'd'.

    result = app.generate_imagenet_wnid_hierarchy(['n12345'], max_depth=10, max_hypernym_depth=2)

    # Expected: {'c': 'd'} (since d is leaf, it becomes string 'd' or list ['d'] depending on post-process)
    # 'd' is a dict key because it's in the path loop, but it has no children in the loop.
    # The loop builds: tree['c']['d'] = {}
    # flatten post process:
    # tree['c']['d'] = {} -> tree['c'] = ['d'] (leaf conversion)

    assert 'a' not in result
    assert 'b' not in result
    assert 'c' in result
    assert 'd' in result['c'] or result['c'] == ['d']

def test_hypernym_depth_full(mock_wn_filtering):
    # Test with hypernym_depth = 0 (Full)
    wnid = 'n12345'
    synset = MagicMock()
    mock_wn_filtering.synset_from_pos_and_offset.return_value = synset

    node_a = MagicMock(); node_a.name.return_value = 'a.n.01'
    node_b = MagicMock(); node_b.name.return_value = 'b.n.01'

    synset.hypernym_paths.return_value = [[node_a, node_b]]

    result = app.generate_imagenet_wnid_hierarchy(['n12345'], max_depth=10, max_hypernym_depth=0)

    assert 'a' in result
    assert 'b' in result['a']
