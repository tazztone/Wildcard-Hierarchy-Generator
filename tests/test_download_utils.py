import pytest
from unittest.mock import patch, MagicMock, mock_open
import os
import download_utils

@pytest.fixture
def mock_urllib():
    with patch('download_utils.urllib.request.urlretrieve') as mock:
        yield mock

@pytest.fixture
def mock_zipfile():
    with patch('download_utils.zipfile.ZipFile') as mock:
        yield mock

def test_download_file_exists(mock_urllib, tmp_path):
    dest = tmp_path / "test.txt"
    dest.touch()

    download_utils.download_file("http://example.com", str(dest))

    mock_urllib.assert_not_called()

def test_download_file_new(mock_urllib, tmp_path):
    dest = tmp_path / "subdir" / "test.txt"

    download_utils.download_file("http://example.com/test.txt", str(dest))

    assert dest.parent.exists()
    mock_urllib.assert_called()

def test_download_file_error(mock_urllib, tmp_path):
    mock_urllib.side_effect = Exception("Network error")
    dest = tmp_path / "test.txt"

    with pytest.raises(Exception, match="Network error"):
        download_utils.download_file("http://example.com", str(dest))

def test_unzip_file(mock_zipfile, tmp_path):
    zip_path = str(tmp_path / "test.zip")
    extract_to = str(tmp_path / "extracted")

    mock_zip_instance = MagicMock()
    mock_zipfile.return_value.__enter__.return_value = mock_zip_instance

    download_utils.unzip_file(zip_path, extract_to)

    mock_zipfile.assert_called_with(zip_path, 'r')
    mock_zip_instance.extractall.assert_called_with(extract_to)

def test_unzip_file_error(mock_zipfile):
    mock_zipfile.side_effect = Exception("Bad zip")
    with pytest.raises(Exception, match="Bad zip"):
        download_utils.unzip_file("test.zip", "out")

def test_ensure_coco_data_exists(tmp_path):
    # Mock existence of json file
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "annotations").mkdir()
    json_path = data_dir / "annotations" / "instances_train2017.json"
    json_path.touch()

    with patch('download_utils.download_file') as mock_dl:
        path = download_utils.ensure_coco_data(str(data_dir))
        assert path == str(json_path)
        mock_dl.assert_not_called()

def test_ensure_coco_data_download(tmp_path, mock_zipfile):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # After unzip, we expect the file to exist.
    # We can side_effect the unzip_file or just manually create it during test
    # But since ensure_coco_data calls unzip_file, we should mock unzip_file logic or the function itself.

    # Let's mock download_file and unzip_file functions directly to test ensure_coco_data logic
    with patch('download_utils.download_file') as mock_dl, \
         patch('download_utils.unzip_file') as mock_unzip, \
         patch('os.remove') as mock_remove:

        # We need to ensure os.path.exists returns True for the json file at the end
        # But we are using real FS for tmp_path.
        # So we should create the file.

        def side_effect_unzip(zip_path, extract_to):
            # Create the file that is expected
            (data_dir / "annotations").mkdir(exist_ok=True)
            (data_dir / "annotations" / "instances_train2017.json").touch()

        mock_unzip.side_effect = side_effect_unzip

        path = download_utils.ensure_coco_data(str(data_dir))

        assert path == str(data_dir / "annotations" / "instances_train2017.json")
        mock_dl.assert_called()
        mock_unzip.assert_called()
        mock_remove.assert_called()

def test_ensure_openimages_data(tmp_path):
    with patch('download_utils.download_file') as mock_dl:
        h, c = download_utils.ensure_openimages_data(str(tmp_path))

        assert h == str(tmp_path / "bbox_labels_600_hierarchy.json")
        assert c == str(tmp_path / "oidv7-class-descriptions.csv")
        assert mock_dl.call_count == 2

def test_ensure_imagenet_list(tmp_path):
    with patch('download_utils.download_file') as mock_dl:
        path = download_utils.ensure_imagenet_list(str(tmp_path))
        assert path == str(tmp_path / "imagenet_class_index.json")
        mock_dl.assert_called_once()

def test_ensure_imagenet21k_data(tmp_path):
    with patch('download_utils.download_file') as mock_dl:
        ids, lemmas = download_utils.ensure_imagenet21k_data(str(tmp_path))
        assert ids == str(tmp_path / "imagenet21k_wordnet_ids.txt")
        assert lemmas == str(tmp_path / "imagenet21k_wordnet_lemmas.txt")
        assert mock_dl.call_count == 2
