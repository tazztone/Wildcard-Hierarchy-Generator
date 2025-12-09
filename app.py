#!/usr/bin/env python3
"""
Hierarchy Generator App

This script generates YAML hierarchies for different datasets (ImageNet, COCO, Open Images).
"""

import argparse
import logging
import os
import sys
import json
import csv
from typing import List, Dict, Optional, Any, Set

import yaml
import nltk
from nltk.corpus import wordnet as wn
from tqdm import tqdm

import download_utils

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# --- Shared Utils ---

def ensure_output_dir(file_path: str) -> None:
    """Ensures the directory for the output file exists."""
    directory = os.path.dirname(file_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

def ensure_nltk_data() -> None:
    """Ensures NLTK WordNet data is available."""
    try:
        wn.ensure_loaded()
    except LookupError:
        logger.info("Downloading WordNet data...")
        try:
            nltk.download('wordnet')
            nltk.download('omw-1.4')
        except Exception as e:
            logger.error(f"Failed to download WordNet data: {e}")
            sys.exit(1)

def save_hierarchy(hierarchy: Any, output_path: str) -> None:
    ensure_output_dir(output_path)
    with open(output_path, 'w') as f:
        yaml.dump(hierarchy, f, sort_keys=False, default_flow_style=False, indent=4)
    logger.info(f"Saved to {output_path}")

def convert_to_wildcard_format(data: Any) -> Any:
    """
    Converts a nested dictionary hierarchy into a format suitable for wildcards.

    Rules:
    1. If a node has only leaf children, it becomes a List of strings.
    2. If a node has mixed children (leaves and subtrees), it becomes a Dict.
       - Subtrees remain as keys.
       - Leaf children are wrapped as `leaf_name: [leaf_name]`.
    """
    # Helper to check if a value represents a leaf in the input format
    def is_leaf_content(val):
        if val == {}: return True  # Legacy empty dict leaf
        if isinstance(val, str): return True  # Tree leaf string
        if val is None: return True
        return False

    # Base case: if data itself is a leaf content (shouldn't happen for root usually, but for recursion)
    if is_leaf_content(data):
        return data

    if isinstance(data, list):
        # Already a list (e.g. from COCO), preserve it
        return data

    if isinstance(data, dict):
        processed_children = {}
        child_is_leaf = {}

        for k, v in data.items():
            converted_v = convert_to_wildcard_format(v)

            # Check what kind of child we have now
            if is_leaf_content(converted_v):
                # It's a leaf. Use the key 'k' or the value 'converted_v' if it's a name string.
                # In legacy, v={}, k=name. In tree, v=name.
                # Let's standardize on the name.
                name = k if (converted_v == {} or converted_v is None) else converted_v
                processed_children[k] = name
                child_is_leaf[k] = True
            elif isinstance(converted_v, list):
                # It's a list of items (a category)
                processed_children[k] = converted_v
                child_is_leaf[k] = False
            elif isinstance(converted_v, dict):
                # It's a subtree
                processed_children[k] = converted_v
                child_is_leaf[k] = False
            else:
                # Fallback
                processed_children[k] = converted_v
                child_is_leaf[k] = False

        # Decide if this node should be a List or Dict
        if not processed_children:
            return {} # Empty dict

        if all(child_is_leaf.values()):
            # All children are leaves -> Return List of names
            return list(processed_children.values())
        else:
            # Mixed or all subtrees -> Return Dict
            result = {}
            for k, val in processed_children.items():
                if child_is_leaf[k]:
                    # Wrap leaf in list
                    result[k] = [val]
                else:
                    result[k] = val
            return result

    return data

# --- Legacy ImageNet Logic (Bottom-Up from WNIDs) ---

def get_synset_from_wnid(wnid: str) -> Optional[Any]:
    try:
        if len(wnid) < 2:
            return None
        pos = wnid[0]
        offset_str = wnid[1:]
        if not offset_str.isdigit():
             return None
        offset = int(offset_str)
        return wn.synset_from_pos_and_offset(pos, offset)
    except Exception as e:
        # logger.error(f"Error finding synset for {wnid}: {e}") # Reduce spam for large lists
        return None

def build_hierarchy_tree_legacy(wnids: List[str]) -> Dict[str, Any]:
    tree: Dict[str, Any] = {}
    for wnid in tqdm(wnids, desc="Building hierarchy", unit="id"):
        synset = get_synset_from_wnid(wnid)
        if not synset:
            continue
        paths = synset.hypernym_paths()
        if not paths:
            continue
        primary_path = paths[0]
        current_level = tree
        for node in primary_path:
            node_name = node.name().split('.')[0]
            key = f"{node_name}"
            if key not in current_level:
                current_level[key] = {}
            current_level = current_level[key]
    return tree

def load_wnids(inputs: List[str]) -> List[str]:
    wnids_to_process = []
    for input_str in inputs:
        if os.path.isfile(input_str):
            try:
                with open(input_str, 'r') as f:
                    lines = [line.strip() for line in f if line.strip()]
                    wnids_to_process.extend(lines)
            except Exception as e:
                logger.error(f"Error reading file {input_str}: {e}")
        else:
            wnids_to_process.append(input_str)
    return list(dict.fromkeys(wnids_to_process))

def generate_imagenet_wnid_hierarchy(wnids: List[str]) -> Dict[str, Any]:
    ensure_nltk_data()
    if not wnids:
        logger.warning("No WNIDs to process.")
        return {}

    logger.info(f"Processing {len(wnids)} IDs (Bottom-Up)...")
    return build_hierarchy_tree_legacy(wnids)

def handle_imagenet_wnid(args) -> None:
    if not args.inputs:
        logger.info("No input provided. Using sample IDs.")
        wnids = ['n02084071', 'n02113799', 'n07753592'] # Simplified sample
    else:
        wnids = load_wnids(args.inputs)

    hierarchy = generate_imagenet_wnid_hierarchy(wnids)
    hierarchy = convert_to_wildcard_format(hierarchy)
    save_hierarchy(hierarchy, args.output)

def handle_imagenet_21k(args) -> None:
    logger.info("Downloading and processing ImageNet-21K data...")
    ids_path, _ = download_utils.ensure_imagenet21k_data()
    logger.info(f"Loading WNIDs from {ids_path}...")
    wnids = load_wnids([ids_path])
    hierarchy = generate_imagenet_wnid_hierarchy(wnids)
    hierarchy = convert_to_wildcard_format(hierarchy)
    save_hierarchy(hierarchy, args.output)

# --- ImageNet Tree Logic (Top-Down Recursive) ---

def load_valid_wnids(json_path: str) -> Set[str]:
    """Loads valid WNIDs from the ImageNet class index JSON."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        # format: {"0": ["n01440764", "tench"], ...}
        valid_wnids = set()
        for key, value in data.items():
            if isinstance(value, list) and len(value) >= 1:
                valid_wnids.add(value[0])
        return valid_wnids
    except Exception as e:
        logger.error(f"Failed to load valid WNIDs: {e}")
        return set()

def build_hierarchy_snippet_style(synset, valid_wnids: Optional[Set[str]], depth=0, max_depth=3):
    if depth > max_depth:
        return []

    name = synset.lemmas()[0].name().replace('_', ' ')
    children = synset.hyponyms()

    if not children:
        # Leaf
        if valid_wnids is not None:
            wnid = f"{synset.pos()}{synset.offset():08d}"
            if wnid not in valid_wnids:
                return None # Filter out
        return name

    child_nodes = {}
    has_valid_children = False
    
    for child in children:
        child_name = child.lemmas()[0].name().replace('_', ' ')
        child_content = build_hierarchy_snippet_style(child, valid_wnids, depth + 1, max_depth)

        if child_content:
            child_nodes[child_name] = child_content
            has_valid_children = True

    if not has_valid_children:
        if valid_wnids is None:
             return name # Treat as leaf if max depth reached or no children
        else:
             # Check if I am valid myself
             wnid = f"{synset.pos()}{synset.offset():08d}"
             if wnid in valid_wnids:
                 return name
             return None

    return child_nodes

def generate_imagenet_tree_hierarchy(root_str: str, depth: int, filter_enabled: bool) -> Dict[str, Any]:
    ensure_nltk_data()
    
    valid_wnids = None
    if filter_enabled:
        logger.info("Ensuring ImageNet list is available...")
        list_path = download_utils.ensure_imagenet_list()
        valid_wnids = load_valid_wnids(list_path)
        logger.info(f"Loaded {len(valid_wnids)} valid WNIDs for filtering.")
    
    try:
        root_synset = wn.synset(root_str)
    except Exception:
        logger.error(f"Could not find root synset: {root_str}")
        return {}

    logger.info(f"Building hierarchy from {root_str} (Top-Down, max_depth={depth})...")

    # We wrap the result in a dict with the root name
    root_name = root_synset.lemmas()[0].name().replace('_', ' ')
    content = build_hierarchy_snippet_style(root_synset, valid_wnids, max_depth=depth)

    if content:
        return {root_name: content}
    else:
        logger.warning("Resulting hierarchy is empty (possibly due to aggressive filtering).")
        return {}

def handle_imagenet_tree(args) -> None:
    hierarchy = generate_imagenet_tree_hierarchy(args.root, args.depth, args.filter)
    hierarchy = convert_to_wildcard_format(hierarchy)
    save_hierarchy(hierarchy, args.output)


# --- COCO Logic ---

def generate_coco_hierarchy() -> Dict[str, Any]:
    # Use local category file if available to avoid large downloads
    if os.path.exists("coco_categories.json"):
        logger.info("Using local coco_categories.json...")
        with open("coco_categories.json", 'r') as f:
            categories = json.load(f)
    else:
        logger.info("Ensuring COCO data is available...")
        json_path = download_utils.ensure_coco_data()
        logger.info(f"Loading {json_path}...")
        with open(json_path, 'r') as f:
            data = json.load(f)
        categories = data['categories']

    logger.info("Processing COCO categories...")
    hierarchy = {}
    for cat in tqdm(categories, desc="Processing COCO"):
        supercat = cat['supercategory']
        name = cat['name']

        if supercat not in hierarchy:
            hierarchy[supercat] = []

        hierarchy[supercat].append(name)
    return hierarchy

def handle_coco(args) -> None:
    hierarchy = generate_coco_hierarchy()
    hierarchy = convert_to_wildcard_format(hierarchy)
    save_hierarchy(hierarchy, args.output)


# --- Open Images Logic ---

def parse_openimages_node(node, id_to_name):
    label_id = node.get('LabelName')
    name = id_to_name.get(label_id, label_id)

    # Handle implicit root
    if label_id == '/m/0bl9f' and name == label_id:
        name = 'Entity'

    # The JSON key is usually 'Subcategory', but let's be safe
    sub_key = None
    if 'Subcategory' in node:
        sub_key = 'Subcategory'
    elif 'Subcategories' in node:
        sub_key = 'Subcategories'

    if sub_key:
        children = {}
        for sub in node[sub_key]:
            child_res = parse_openimages_node(sub, id_to_name)
            if isinstance(child_res, dict):
                children.update(child_res)
            else:
                if 'misc' not in children:
                    children['misc'] = []
                children['misc'].append(child_res)
        return {name: children}
    else:
        return name

def generate_openimages_hierarchy() -> Dict[str, Any]:
    logger.info("Ensuring Open Images data is available...")
    hierarchy_path, classes_path = download_utils.ensure_openimages_data()

    logger.info("Loading class descriptions...")
    id_to_name = {}
    with open(classes_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                id_to_name[row[0]] = row[1]

    logger.info("Loading hierarchy JSON...")
    with open(hierarchy_path, 'r') as f:
        data = json.load(f)

    logger.info("Building hierarchy...")
    return parse_openimages_node(data, id_to_name)

def handle_openimages(args) -> None:
    hierarchy = generate_openimages_hierarchy()
    hierarchy = convert_to_wildcard_format(hierarchy)
    save_hierarchy(hierarchy, args.output)


# --- Main ---

def main() -> None:
    parser = argparse.ArgumentParser(description="Wildcard Hierarchy Generator")
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Legacy ImageNet (List of IDs -> Hierarchy)
    p_wnid = subparsers.add_parser('imagenet-wnid', help='Build hierarchy from list of WNIDs (Bottom-Up)')
    p_wnid.add_argument('inputs', nargs='*', help="WNIDs or file paths")
    p_wnid.add_argument('-o', '--output', default='imagenet_hierarchy.yaml')
    p_wnid.add_argument('-v', '--verbose', action='store_true')
    p_wnid.set_defaults(func=handle_imagenet_wnid)

    # ImageNet-21K
    p_21k = subparsers.add_parser('imagenet-21k', help='Build hierarchy for ImageNet-21K')
    p_21k.add_argument('-o', '--output', default='imagenet21k_hierarchy.yaml')
    p_21k.set_defaults(func=handle_imagenet_21k)

    # New ImageNet (Root -> Recursive Children)
    p_tree = subparsers.add_parser('imagenet-tree', help='Build hierarchy recursively from a root node (Top-Down)')
    p_tree.add_argument('--root', default='animal.n.01', help="Root synset (default: animal.n.01)")
    p_tree.add_argument('--depth', type=int, default=3, help="Max recursion depth")
    p_tree.add_argument('--filter', action='store_true', help="Filter using ImageNet 1k list")
    p_tree.add_argument('-o', '--output', default='wildcards_imagenet.yaml')
    p_tree.set_defaults(func=handle_imagenet_tree)

    # COCO
    p_coco = subparsers.add_parser('coco', help='Build hierarchy from COCO annotations')
    p_coco.add_argument('-o', '--output', default='wildcards_coco.yaml')
    p_coco.set_defaults(func=handle_coco)

    # Open Images
    p_oi = subparsers.add_parser('openimages', help='Build hierarchy from Open Images data')
    p_oi.add_argument('-o', '--output', default='wildcards_openimages.yaml')
    p_oi.set_defaults(func=handle_openimages)

    args = parser.parse_args()

    if hasattr(args, 'verbose') and args.verbose:
        logger.setLevel(logging.DEBUG)

    args.func(args)

if __name__ == "__main__":
    main()
