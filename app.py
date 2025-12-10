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

DOWNLOADS_DIR = "downloads"

# Categories to optionally blacklist
ABSTRACT_CATEGORIES = {
    'entity', 'abstraction', 'communication', 'measure',
    'attribute', 'state', 'event', 'act', 'group',
    'relation', 'possession', 'phenomenon'
}

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
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(hierarchy, f, sort_keys=False, default_flow_style=False, indent=4)
    logger.info(f"Saved to {output_path}")

def convert_to_wildcard_format(data: Any) -> Any:
    """
    Converts a nested dictionary hierarchy into a format suitable for wildcards.
    """
    def is_leaf_content(val):
        if val == {}: return True
        if isinstance(val, str): return True
        if val is None: return True
        return False

    if is_leaf_content(data):
        return data

    if isinstance(data, list):
        return sorted(list(set(data))) # Deduplicate and sort

    if isinstance(data, dict):
        processed_children = {}
        child_is_leaf = {}

        for k, v in data.items():
            converted_v = convert_to_wildcard_format(v)
            if is_leaf_content(converted_v):
                name = k if (converted_v == {} or converted_v is None) else converted_v
                processed_children[k] = name
                child_is_leaf[k] = True
            elif isinstance(converted_v, list):
                # Check for redundancy: key == item[0] and len==1
                if len(converted_v) == 1 and converted_v[0] == k:
                    processed_children[k] = converted_v[0]
                    child_is_leaf[k] = True
                else:
                    processed_children[k] = converted_v
                    child_is_leaf[k] = False
            elif isinstance(converted_v, dict):
                processed_children[k] = converted_v
                child_is_leaf[k] = False
            else:
                processed_children[k] = converted_v
                child_is_leaf[k] = False

        if not processed_children:
            return {}

        if all(child_is_leaf.values()):
            return sorted(list(processed_children.values()))
        else:
            result = {}
            for k, val in processed_children.items():
                if child_is_leaf[k]:
                    result[k] = [val]
                else:
                    result[k] = val
            return result

    return data

def extract_all_leaves(data: Any) -> List[str]:
    """Helper to recursively extract all strings from a nested structure."""
    leaves = []
    if isinstance(data, list):
        for item in data:
            leaves.extend(extract_all_leaves(item))
    elif isinstance(data, dict):
        for k, v in data.items():
            if v == {} or v is None:
                leaves.append(k)
            else:
                leaves.extend(extract_all_leaves(v))
    elif isinstance(data, str):
        leaves.append(data)
    elif data is None:
        pass
    return leaves

def flatten_hierarchy_post_process(data: Any, current_depth: int = 0, max_depth: int = 10) -> Any:
    """
    Generic post-processor to flatten hierarchy at max_depth.
    When max_depth is reached, all descendants are collected into a flat list.
    """
    if current_depth >= max_depth:
        return extract_all_leaves(data)

    if isinstance(data, dict):
        new_dict = {}
        for k, v in data.items():
            new_dict[k] = flatten_hierarchy_post_process(v, current_depth + 1, max_depth)
        return new_dict

    # Lists are usually leaves or categories, but if we have list of dicts (not expected in our format), handle it?
    # In our format, lists are usually strings.
    return data

# --- Data Loaders ---

def load_wnids_list(inputs: List[str]) -> List[str]:
    wnids_to_process = []
    for input_str in inputs:
        if os.path.isfile(input_str):
            try:
                with open(input_str, 'r', encoding='utf-8') as f:
                    lines = [line.strip() for line in f if line.strip()]
                    wnids_to_process.extend(lines)
            except Exception as e:
                logger.error(f"Error reading file {input_str}: {e}")
        else:
            wnids_to_process.append(input_str)
    return list(dict.fromkeys(wnids_to_process))

def load_imagenet_1k_set() -> Set[str]:
    logger.info("Ensuring ImageNet 1k list is available...")
    list_path = download_utils.ensure_imagenet_list(DOWNLOADS_DIR)
    try:
        with open(list_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        valid_wnids = set()
        for key, value in data.items():
            if isinstance(value, list) and len(value) >= 1:
                valid_wnids.add(value[0])
        return valid_wnids
    except Exception as e:
        logger.error(f"Failed to load valid WNIDs: {e}")
        return set()

def load_imagenet_21k_set() -> Set[str]:
    logger.info("Ensuring ImageNet 21k list is available...")
    ids_path, _ = download_utils.ensure_imagenet21k_data(DOWNLOADS_DIR)
    wnids = set()
    try:
        with open(ids_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    wnids.add(line)
    except Exception as e:
        logger.error(f"Failed to load 21k WNIDs: {e}")
    return wnids

# --- ImageNet WNID (Bottom-Up) ---

def get_synset_from_wnid(wnid: str) -> Optional[Any]:
    try:
        if len(wnid) < 2: return None
        pos = wnid[0]
        offset = int(wnid[1:])
        return wn.synset_from_pos_and_offset(pos, offset)
    except Exception:
        return None

def generate_imagenet_wnid_hierarchy(wnids: List[str], max_depth: int = 10, max_hypernym_depth: Optional[int] = None) -> Dict[str, Any]:
    ensure_nltk_data()
    if not wnids: return {}

    logger.info(f"Processing {len(wnids)} IDs (Bottom-Up)...")
    tree: Dict[str, Any] = {}

    for wnid in tqdm(wnids, desc="Building hierarchy", unit="id"):
        synset = get_synset_from_wnid(wnid)
        if not synset: continue
        paths = synset.hypernym_paths()
        if not paths: continue

        # Use primary path
        primary_path = paths[0]

        # Apply hypernym depth limit if set
        if max_hypernym_depth is not None and max_hypernym_depth > 0:
            # Take last N items from path to limit height
            path_to_use = primary_path[-max_hypernym_depth:]
        else:
            path_to_use = primary_path

        current_level = tree
        for node in path_to_use:
            node_name = node.name().split('.')[0] # e.g. 'dog' from 'dog.n.01'
            if node_name not in current_level:
                current_level[node_name] = {}
            current_level = current_level[node_name]

    # Flatten based on max_depth from root
    return flatten_hierarchy_post_process(tree, 0, max_depth)

# --- ImageNet Tree (Top-Down Recursive) ---

def get_primary_synset(word: str) -> Optional[Any]:
    """Get only the first (most common) synset"""
    try:
        synsets = wn.synsets(word.replace(' ', '_'))
        if synsets:
            return synsets[0]  # Only use first/primary meaning
    except Exception:
        pass
    return None

def get_all_descendants(synset, valid_wnids: Optional[Set[str]]) -> List[str]:
    """Fetches all descendants recursively, filtering if valid_wnids is present."""
    descendants = set()
    try:
        # closure is a generator
        iterator = synset.closure(lambda s: s.hyponyms())
        for s in iterator:
            wnid = f"{s.pos()}{s.offset():08d}"
            name = s.lemmas()[0].name().replace('_', ' ')
            if valid_wnids:
                if wnid in valid_wnids:
                    descendants.add(name)
            else:
                descendants.add(name)
    except Exception as e:
        logger.warning(f"Error traversing descendants of {synset}: {e}")

    return sorted(list(descendants))

def build_hierarchy_tree_recursive(synset, valid_wnids: Optional[Set[str]], depth, max_depth, strict_filter: bool = True, blacklist: bool = False):
    name = synset.lemmas()[0].name().replace('_', ' ')

    # 1. Blacklist check (Solution 3)
    if blacklist:
        node_lemma = synset.name().split('.')[0]
        if node_lemma in ABSTRACT_CATEGORIES:
            return None

    # 2. Strict Primary Synset Check (Solution 1)
    if strict_filter:
        primary = get_primary_synset(name)
        if primary and primary != synset:
             # This synset is not the primary meaning for its name. Skip it.
             return None

    # Check if we should stop and flatten
    if depth >= max_depth:
        # Flatten: Get all descendants
        leaves = get_all_descendants(synset, valid_wnids)
        if leaves:
             return leaves # Return list of strings
        else:
             # If no descendants, but I am a node... return myself?
             # Or return empty? "Includes all words... just flattened".
             # If I have no children, I am the leaf.
             # Check if I am valid
             if valid_wnids:
                 wnid = f"{synset.pos()}{synset.offset():08d}"
                 if wnid in valid_wnids:
                     return [name] # Return self as a leaf item
                 return []
             return [name]

    children = synset.hyponyms()

    child_nodes = {}
    has_valid_children = False
    
    for child in children:
        child_name = child.lemmas()[0].name().replace('_', ' ')
        child_content = build_hierarchy_tree_recursive(child, valid_wnids, depth + 1, max_depth, strict_filter, blacklist)

        if child_content:
            # If child_content is a list of strings (flattened leaves)
            child_nodes[child_name] = child_content
            has_valid_children = True

    if not has_valid_children:
        # I am a leaf relative to the traversal or filtration
        if valid_wnids:
            wnid = f"{synset.pos()}{synset.offset():08d}"
            if wnid not in valid_wnids:
                return None
        return name # Return string as leaf indicator (legacy style used by convert_to_wildcard)

    return child_nodes

def generate_imagenet_tree_hierarchy(root_str: str, max_depth: int, filter_ids: Optional[Set[str]], strict_filter: bool = True, blacklist: bool = False) -> Dict[str, Any]:
    ensure_nltk_data()
    
    if not root_str:
        root_str = 'entity.n.01'

    try:
        root_synset = wn.synset(root_str)
    except Exception:
        logger.error(f"Could not find root synset: {root_str}")
        return {}

    logger.info(f"Building hierarchy from {root_str} (Depth={max_depth}, Strict={strict_filter}, Blacklist={blacklist})...")

    root_name = root_synset.lemmas()[0].name().replace('_', ' ')

    # Check strict/blacklist for root itself? Usually root is explicit so we keep it.
    # But pass flags to children.
    content = build_hierarchy_tree_recursive(root_synset, filter_ids, 0, max_depth, strict_filter, blacklist)

    if content:
        return {root_name: content}
    return {}

# --- COCO Logic ---

def generate_coco_hierarchy(max_depth: int = 10) -> Dict[str, Any]:
    local_coco_path = os.path.join(DOWNLOADS_DIR, "coco_categories.json")
    if os.path.exists(local_coco_path):
        with open(local_coco_path, 'r', encoding='utf-8') as f:
            categories = json.load(f)
    else:
        json_path = download_utils.ensure_coco_data(DOWNLOADS_DIR)
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        categories = data['categories']

    hierarchy = {}
    for cat in tqdm(categories, desc="Processing COCO"):
        supercat = cat['supercategory']
        name = cat['name']
        if supercat not in hierarchy:
            hierarchy[supercat] = []
        hierarchy[supercat].append(name)

    return flatten_hierarchy_post_process(hierarchy, 0, max_depth)

# --- Open Images Logic ---

def parse_openimages_node(node, id_to_name):
    label_id = node.get('LabelName')
    name = id_to_name.get(label_id, label_id)
    if label_id == '/m/0bl9f' and name == label_id:
        name = 'Entity'

    sub_key = 'Subcategory' if 'Subcategory' in node else ('Subcategories' if 'Subcategories' in node else None)

    if sub_key:
        children = {}
        for sub in node[sub_key]:
            child_res = parse_openimages_node(sub, id_to_name)
            if isinstance(child_res, dict):
                children.update(child_res)
            else:
                if 'misc' not in children: children['misc'] = []
                children['misc'].append(child_res)
        return {name: children}
    else:
        return name

def generate_openimages_hierarchy(max_depth: int = 10) -> Dict[str, Any]:
    hierarchy_path, classes_path = download_utils.ensure_openimages_data(DOWNLOADS_DIR)

    id_to_name = {}
    with open(classes_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2: id_to_name[row[0]] = row[1]

    with open(hierarchy_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    full_hierarchy = parse_openimages_node(data, id_to_name)
    return flatten_hierarchy_post_process(full_hierarchy, 0, max_depth)


# --- CLI Handlers ---

def handle_imagenet_wnid(args):
    if not args.inputs:
        wnids = ['n02084071', 'n02113799', 'n07753592']
    else:
        wnids = load_wnids_list(args.inputs)
    h = generate_imagenet_wnid_hierarchy(wnids, args.depth, args.hypernym_depth)
    save_hierarchy(convert_to_wildcard_format(h), args.output)

def handle_imagenet_tree(args):
    filter_set = None
    if args.filter == '1k':
        filter_set = load_imagenet_1k_set()
    elif args.filter == '21k':
        filter_set = load_imagenet_21k_set()

    h = generate_imagenet_tree_hierarchy(args.root, args.depth, filter_set, not args.no_strict, args.blacklist)
    save_hierarchy(convert_to_wildcard_format(h), args.output)

def handle_coco(args):
    h = generate_coco_hierarchy(args.depth)
    save_hierarchy(convert_to_wildcard_format(h), args.output)

def handle_openimages(args):
    h = generate_openimages_hierarchy(args.depth)
    save_hierarchy(convert_to_wildcard_format(h), args.output)

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', required=True)

    # ImageNet WNID
    p_wnid = subparsers.add_parser('imagenet-wnid')
    p_wnid.add_argument('inputs', nargs='*')
    p_wnid.add_argument('-o', '--output', default='imagenet_hierarchy.yaml')
    p_wnid.add_argument('--depth', type=int, default=10)
    p_wnid.add_argument('--hypernym-depth', type=int, default=None, help='Limit depth of hypernym path (Bottom-Up)')
    p_wnid.set_defaults(func=handle_imagenet_wnid)

    # ImageNet Tree (Unified)
    p_tree = subparsers.add_parser('imagenet-tree')
    p_tree.add_argument('--root', default='entity.n.01')
    p_tree.add_argument('--depth', type=int, default=3)
    p_tree.add_argument('--filter', choices=['1k', '21k', 'none'], default='none')
    p_tree.add_argument('--no-strict', action='store_true', help='Disable strict primary synset filtering')
    p_tree.add_argument('--blacklist', action='store_true', help='Blacklist abstract categories')
    p_tree.add_argument('-o', '--output', default='wildcards_imagenet.yaml')
    p_tree.set_defaults(func=handle_imagenet_tree)

    # COCO
    p_coco = subparsers.add_parser('coco')
    p_coco.add_argument('-o', '--output', default='wildcards_coco.yaml')
    p_coco.add_argument('--depth', type=int, default=10)
    p_coco.set_defaults(func=handle_coco)

    # Open Images
    p_oi = subparsers.add_parser('openimages')
    p_oi.add_argument('-o', '--output', default='wildcards_openimages.yaml')
    p_oi.add_argument('--depth', type=int, default=10)
    p_oi.set_defaults(func=handle_openimages)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
