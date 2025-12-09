import pytest
from unittest.mock import patch, MagicMock, mock_open
import sys
import app
import json
import logging

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

    # Path: entity -> animal -> dog
    mock.hypernym_paths.return_value = [[mock_entity, mock_animal, mock]]

    # Children for top-down
    mock.hyponyms.return_value = []

    return mock

@pytest.fixture
def mock_wn_fixture(mock_synset):
    """
    Mocks app.wn safely by replacing the object directly.
    """
    original_wn = app.wn
    mock_wn = MagicMock()
    app.wn = mock_wn

    # Setup default behaviors
    mock_wn.synset_from_pos_and_offset.return_value = mock_synset

    yield mock_wn

    app.wn = original_wn

def test_get_synset_from_wnid_valid(mock_wn_fixture, mock_synset):
    # Setup specific return for this test if needed, or use fixture default
    result = app.get_synset_from_wnid('n02084071')
    assert result == mock_synset
    mock_wn_fixture.synset_from_pos_and_offset.assert_called_with('n', 2084071)

def test_get_synset_from_wnid_invalid_format(mock_wn_fixture):
    result = app.get_synset_from_wnid('invalid')
    assert result is None

def test_get_synset_from_wnid_invalid_offset(mock_wn_fixture):
    result = app.get_synset_from_wnid('nABC')
    assert result is None

def test_get_synset_from_wnid_exception(mock_wn_fixture):
    mock_wn_fixture.synset_from_pos_and_offset.side_effect = Exception("Boom")
    result = app.get_synset_from_wnid('n02084071')
    assert result is None

def test_build_hierarchy_tree_legacy(mock_wn_fixture, mock_synset):
    # Ensure get_synset_from_wnid calls our mock
    # We don't mock get_synset_from_wnid here, we rely on mock_wn_fixture making it work

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
    p.write_text("n11111111\nn22222222", encoding='utf-8')

    inputs = [str(p), 'n33333333']
    result = app.load_wnids(inputs)

    assert len(result) == 3
    assert 'n11111111' in result

def test_load_wnids_exception(caplog):
    # Pass a file that exists but raises error on read (permission?)
    with patch('builtins.open', mock_open()) as mock_file:
        mock_file.side_effect = Exception("Read Error")
        # We need os.path.isfile to be true
        with patch('os.path.isfile', return_value=True):
             result = app.load_wnids(['bad_file.txt'])

    assert "Error reading file bad_file.txt: Read Error" in caplog.text
    assert result == []

def test_coco_logic(tmp_path):
    coco_data = {
        "categories": [
            {"id": 1, "name": "bicycle", "supercategory": "vehicle"},
            {"id": 2, "name": "car", "supercategory": "vehicle"},
            {"id": 3, "name": "dog", "supercategory": "animal"}
        ]
    }

    json_file = tmp_path / "instances.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f)

    with patch('app.download_utils.ensure_coco_data') as mock_ensure:
        mock_ensure.return_value = str(json_file)

        args = MagicMock()
        args.output = str(tmp_path / "coco_out.yaml")

        app.handle_coco(args)

        import yaml
        with open(args.output, 'r', encoding='utf-8') as f:
             output = yaml.safe_load(f)

        assert "vehicle" in output
        assert "bicycle" in output["vehicle"]
        assert "car" in output["vehicle"]
        assert "animal" in output
        assert "dog" in output["animal"]

def test_openimages_logic(tmp_path):
    csv_file = tmp_path / "classes.csv"
    with open(csv_file, 'w', encoding='utf-8') as f:
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
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(hierarchy_data, f)

    with patch('app.download_utils.ensure_openimages_data') as mock_ensure:
        mock_ensure.return_value = (str(json_file), str(csv_file))

        args = MagicMock()
        args.output = str(tmp_path / "oi_out.yaml")

        app.handle_openimages(args)

        import yaml
        with open(args.output, 'r', encoding='utf-8') as f:
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

def test_handle_imagenet_tree(mock_wn_fixture, tmp_path):
    root_synset = MagicMock()
    root_lemma = MagicMock()
    root_lemma.name.return_value = 'animal'
    root_synset.lemmas.return_value = [root_lemma]
    root_synset.hyponyms.return_value = [] # Leaf

    mock_wn_fixture.synset.return_value = root_synset

    args = MagicMock()
    args.root = 'animal.n.01'
    args.depth = 1
    args.filter = False
    args.output = str(tmp_path / "tree.yaml")

    app.handle_imagenet_tree(args)

    mock_wn_fixture.synset.assert_called_with('animal.n.01')

    import yaml
    with open(args.output, 'r', encoding='utf-8') as f:
        output = yaml.safe_load(f)

    assert output == ['animal']

def test_handle_imagenet_wnid(mock_wn_fixture, tmp_path):
    args = MagicMock()
    args.inputs = [] # Sample
    args.output = str(tmp_path / "wnid.yaml")

    app.handle_imagenet_wnid(args)

    import yaml
    with open(args.output, 'r', encoding='utf-8') as f:
        output = yaml.safe_load(f)

    assert 'entity' in output
    assert 'animal' in output['entity']

def test_ensure_nltk_data(mock_wn_fixture):
    # Test ensure_loaded is called
    app.ensure_nltk_data()
    mock_wn_fixture.ensure_loaded.assert_called()

def test_ensure_nltk_data_download_fail(caplog):
    original_wn = app.wn
    mock_wn = MagicMock()
    mock_wn.ensure_loaded.side_effect = LookupError
    app.wn = mock_wn

    try:
        with patch('app.nltk.download', side_effect=Exception("Download failed")):
            with pytest.raises(SystemExit):
                app.ensure_nltk_data()
    finally:
        app.wn = original_wn

    assert "Failed to download WordNet data: Download failed" in caplog.text

def test_handle_imagenet_21k(mock_wn_fixture, tmp_path):
    ids_file = tmp_path / "imagenet21k_wordnet_ids.txt"
    ids_file.write_text("n00001234", encoding='utf-8')

    with patch('app.download_utils.ensure_imagenet21k_data') as mock_ensure:
        mock_ensure.return_value = (str(ids_file), "dummy_lemmas")

        args = MagicMock()
        args.output = str(tmp_path / "21k.yaml")

        app.handle_imagenet_21k(args)

        mock_ensure.assert_called()

        import yaml
        with open(args.output, 'r', encoding='utf-8') as f:
            output = yaml.safe_load(f)
            assert output

def test_load_valid_wnids_error(caplog):
    with patch('builtins.open', side_effect=Exception("JSON error")):
        result = app.load_valid_wnids("bad.json")

    assert result == set()
    assert "Failed to load valid WNIDs: JSON error" in caplog.text

def test_main_cli(mock_wn_fixture, tmp_path):
    # Test argparse indirectly by invoking main with patched sys.argv
    output_file = tmp_path / "out.yaml"
    with patch('sys.argv', ['app.py', 'imagenet-wnid', 'n02084071', '-o', str(output_file)]):
        app.main()

    assert output_file.exists()
