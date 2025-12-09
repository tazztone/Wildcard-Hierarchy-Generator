import os
import logging
import urllib.request
import zipfile
import shutil
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

def download_file(url: str, dest_path: str, force: bool = False) -> None:
    """
    Downloads a file from a URL to a destination path.
    """
    if os.path.exists(dest_path) and not force:
        logger.info(f"File already exists: {dest_path}")
        return

    logger.info(f"Downloading {url} to {dest_path}...")
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        urllib.request.urlretrieve(url, dest_path)
        logger.info(f"Download complete: {dest_path}")
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        raise e

def unzip_file(zip_path: str, extract_to: str) -> None:
    """
    Unzips a file to a directory.
    """
    logger.info(f"Extracting {zip_path} to {extract_to}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        logger.info(f"Extraction complete.")
    except Exception as e:
        logger.error(f"Failed to unzip {zip_path}: {e}")
        raise e

def ensure_coco_data(data_dir: str = ".") -> str:
    """
    Ensures COCO annotations are present.
    Returns path to instances_train2017.json.
    """
    zip_name = "annotations_trainval2017.zip"
    zip_path = os.path.join(data_dir, zip_name)
    # The zip extracts to a folder named 'annotations'
    json_path = os.path.join(data_dir, "annotations", "instances_train2017.json")
    url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

    if os.path.exists(json_path):
        return json_path

    if not os.path.exists(zip_path):
        download_file(url, zip_path)

    unzip_file(zip_path, data_dir)

    if not os.path.exists(json_path):
         raise FileNotFoundError(f"Expected {json_path} after extraction but not found.")

    return json_path

def ensure_openimages_data(data_dir: str = ".") -> Tuple[str, str]:
    """
    Ensures Open Images hierarchy and class descriptions are present.
    Returns (hierarchy_path, classes_path).
    """
    hierarchy_url = "https://storage.googleapis.com/openimages/2018_04/bbox_labels_600_hierarchy.json"
    classes_url = "https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv"

    hierarchy_path = os.path.join(data_dir, "bbox_labels_600_hierarchy.json")
    classes_path = os.path.join(data_dir, "class-descriptions-boxable.csv")

    download_file(hierarchy_url, hierarchy_path)
    download_file(classes_url, classes_path)

    return hierarchy_path, classes_path

def ensure_imagenet_list(data_dir: str = ".") -> str:
    """
    Ensures ImageNet 1k list is present.
    Returns path to the file.
    """
    # Maps index to [WNID, label]
    url = "https://raw.githubusercontent.com/raghakot/keras-vis/master/resources/imagenet_class_index.json"
    path = os.path.join(data_dir, "imagenet_class_index.json")

    download_file(url, path)

    return path
