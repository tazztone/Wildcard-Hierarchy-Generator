import pytest
from unittest.mock import patch, MagicMock, mock_open
import app
import json

# Mock data for testing
MOCK_WNID = 'n02084071' # Dog
MOCK_SYNSET_NAME = 'dog.n.01'

@pytest.fixture
def mock_synset():
    mock = MagicMock()
    # Mocking lemma.name()
    lemma = MagicMock()
    lemma.name.return_value = 'dog'
    mock.lemmas.return_value = [lemma]

    mock.name.return_value = MOCK_SYNSET_NAME
    mock.offset.return_value = 2084071
    mock.pos.return_value = 'n'

    # Mock hypernym paths
    mock_entity = MagicMock()
    mock_entity.name.return_value = 'entity.n.01'
    mock_entity.offset.return_value = 1740

    mock_animal = MagicMock()
    mock_animal.name.return_value = 'animal.n.01'
    mock_animal.offset.return_value = 15328

    mock.hypernym_paths.return_value = [[mock_entity, mock_animal, mock]]
    return mock

def test_get_synset_from_wnid_valid():
    # Patch app.wn. The name 'wn' in 'app' module.
    with patch('app.wn') as mock_wn:
        mock_synset_obj = MagicMock()
        mock_wn.synset_from_pos_and_offset.return_value = mock_synset_obj

        result = app.get_synset_from_wnid('n02084071')

        assert result == mock_synset_obj
        mock_wn.synset_from_pos_and_offset.assert_called_with('n', 2084071)

def test_get_synset_from_wnid_invalid_format():
    with patch('app.wn'):
        result = app.get_synset_from_wnid('invalid')
        assert result is None

def test_get_synset_from_wnid_invalid_offset():
    with patch('app.wn'):
        result = app.get_synset_from_wnid('nABC')
        assert result is None

def test_build_hierarchy_tree_legacy(mock_synset):
    with patch('app.get_synset_from_wnid') as mock_get_synset:
        mock_get_synset.return_value = mock_synset

        wnids = ['n02084071']
        tree = app.build_hierarchy_tree_legacy(wnids)

        assert 'entity' in tree
        assert 'animal' in tree['entity']
        assert 'dog' in tree['entity']['animal']
        assert tree['entity']['animal']['dog'] == {}

def test_load_wnids_direct():
    inputs = ['n12345678', 'n87654321']
    result = app.load_wnids(inputs)
    assert result == inputs

def test_load_wnids_file(tmp_path):
    p = tmp_path / "wnids.txt"
    p.write_text("n11111111\nn22222222")

    inputs = [str(p), 'n33333333']
    result = app.load_wnids(inputs)

    assert len(result) == 3

# --- New Tests ---

def test_coco_logic(tmp_path):
    coco_data = {
        "categories": [
            {"id": 1, "name": "bicycle", "supercategory": "vehicle"},
            {"id": 2, "name": "car", "supercategory": "vehicle"},
            {"id": 3, "name": "dog", "supercategory": "animal"}
        ]
    }

    json_file = tmp_path / "instances.json"
    with open(json_file, 'w') as f:
        json.dump(coco_data, f)

    with patch('app.download_utils.ensure_coco_data') as mock_ensure:
        mock_ensure.return_value = str(json_file)

        args = MagicMock()
        args.output = str(tmp_path / "coco_out.yaml")

        app.handle_coco(args)

        import yaml
        with open(args.output, 'r') as f:
             output = yaml.safe_load(f)

        assert "vehicle" in output
        assert "bicycle" in output["vehicle"]
        assert "car" in output["vehicle"]
        assert "animal" in output
        assert "dog" in output["animal"]

def test_openimages_logic(tmp_path):
    csv_file = tmp_path / "classes.csv"
    with open(csv_file, 'w') as f:
        f.write("/m/01g317,Person\n")
        f.write("/m/0199g,Bicycle\n")

    hierarchy_data = {
        "LabelName": "/m/01g317",
        "Subcategories": [
             {
                 "LabelName": "/m/0199g"
             }
        ]
    }
    json_file = tmp_path / "hierarchy.json"
    with open(json_file, 'w') as f:
        json.dump(hierarchy_data, f)

    with patch('app.download_utils.ensure_openimages_data') as mock_ensure:
        mock_ensure.return_value = (str(json_file), str(csv_file))

        args = MagicMock()
        args.output = str(tmp_path / "oi_out.yaml")

        app.handle_openimages(args)

        import yaml
        with open(args.output, 'r') as f:
            output = yaml.safe_load(f)

        assert "Person" in output
        # Based on logic: leaf node returns name string.
        # Subcategory loop: if child result is dict, update children.
        # If child result is string (leaf), append to children['misc'].
        # So Person -> {misc: [Bicycle]}
        assert "misc" in output["Person"]
        assert "Bicycle" in output["Person"]["misc"]

def test_imagenet_tree_logic_basic():
    # Setup mock synset manually to ensure cleanliness
    mock_synset = MagicMock()
    lemma = MagicMock()
    lemma.name.return_value = 'dog'
    mock_synset.lemmas.return_value = [lemma]
    mock_synset.hyponyms.return_value = [] # Leaf

    result = app.build_hierarchy_snippet_style(mock_synset, valid_wnids=None)
    assert result == 'dog'

    # Test with children
    child = MagicMock()
    child_lemma = MagicMock()
    child_lemma.name.return_value = 'puppy'
    child.lemmas.return_value = [child_lemma]
    child.hyponyms.return_value = []

    mock_synset.hyponyms.return_value = [child]

    result = app.build_hierarchy_snippet_style(mock_synset, valid_wnids=None)
    assert result == {'puppy': 'puppy'}

def test_imagenet_tree_logic_filtered():
    mock_synset = MagicMock()
    lemma = MagicMock()
    lemma.name.return_value = 'dog'
    mock_synset.lemmas.return_value = [lemma]
    mock_synset.hyponyms.return_value = []
    mock_synset.pos.return_value = 'n'
    mock_synset.offset.return_value = 2084071

    valid = {'n02084071'}
    result = app.build_hierarchy_snippet_style(mock_synset, valid_wnids=valid)
    assert result == 'dog'

    # If not in valid
    result = app.build_hierarchy_snippet_style(mock_synset, valid_wnids={'n99999999'})
    assert result is None

def test_imagenet_tree_logic_filtered_branch():
    # Root (invalid) -> Child (valid)
    mock_root = MagicMock()
    root_lemma = MagicMock()
    root_lemma.name.return_value = 'root'
    mock_root.lemmas.return_value = [root_lemma]
    mock_root.pos.return_value = 'n'
    mock_root.offset.return_value = 1000

    mock_child = MagicMock()
    child_lemma = MagicMock()
    child_lemma.name.return_value = 'child'
    mock_child.lemmas.return_value = [child_lemma]
    mock_child.hyponyms.return_value = [] # Leaf
    mock_child.pos.return_value = 'n'
    mock_child.offset.return_value = 2000 # Valid ID

    mock_root.hyponyms.return_value = [mock_child]

    # Filter with only child valid
    valid = {'n00002000'}
    result = app.build_hierarchy_snippet_style(mock_root, valid_wnids=valid)

    # Should keep root and child
    assert result == {'child': 'child'}

    # Filter with neither valid
    valid = {'n00009999'}
    result = app.build_hierarchy_snippet_style(mock_root, valid_wnids=valid)
    assert result is None
