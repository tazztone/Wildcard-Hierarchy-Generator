import pytest
from unittest.mock import patch, MagicMock
import app

# Mock data for testing
MOCK_WNID = 'n02084071' # Dog
MOCK_SYNSET_NAME = 'dog.n.01'

@pytest.fixture
def mock_synset():
    mock = MagicMock()
    mock.name.return_value = MOCK_SYNSET_NAME
    mock.offset.return_value = 2084071
    # Mock hypernym paths
    # A simplified path: entity -> ... -> dog
    mock_entity = MagicMock()
    mock_entity.name.return_value = 'entity.n.01'
    mock_entity.offset.return_value = 1740

    mock_animal = MagicMock()
    mock_animal.name.return_value = 'animal.n.01'
    mock_animal.offset.return_value = 15328

    mock.hypernym_paths.return_value = [[mock_entity, mock_animal, mock]]
    return mock

def test_get_synset_from_wnid_valid():
    # Patch the 'wn' object in 'app' module directly
    with patch('app.wn') as mock_wn:
        mock_synset_obj = MagicMock()
        mock_wn.synset_from_pos_and_offset.return_value = mock_synset_obj

        result = app.get_synset_from_wnid('n02084071')

        assert result == mock_synset_obj
        mock_wn.synset_from_pos_and_offset.assert_called_with('n', 2084071)

def test_get_synset_from_wnid_invalid_format():
    result = app.get_synset_from_wnid('invalid')
    assert result is None

def test_get_synset_from_wnid_invalid_offset():
    result = app.get_synset_from_wnid('nABC')
    assert result is None

def test_build_hierarchy_tree(mock_synset):
    with patch('app.get_synset_from_wnid') as mock_get_synset:
        mock_get_synset.return_value = mock_synset

        wnids = ['n02084071']
        tree = app.build_hierarchy_tree(wnids)

        # Expected structure based on mock_synset path: entity -> animal -> dog
        assert 'entity' in tree
        assert 'animal' in tree['entity']
        assert 'dog' in tree['entity']['animal']
        assert tree['entity']['animal']['dog'] == {}

def test_load_wnids_direct():
    inputs = ['n12345678', 'n87654321']
    result = app.load_wnids(inputs)
    assert result == inputs

def test_load_wnids_file(tmp_path):
    # Create a temporary file
    p = tmp_path / "wnids.txt"
    p.write_text("n11111111\nn22222222")

    inputs = [str(p), 'n33333333']
    result = app.load_wnids(inputs)

    assert len(result) == 3
    assert 'n11111111' in result
    assert 'n22222222' in result
    assert 'n33333333' in result

def test_load_wnids_deduplication():
    inputs = ['n111', 'n111']
    result = app.load_wnids(inputs)
    assert len(result) == 1
    assert result[0] == 'n111'
