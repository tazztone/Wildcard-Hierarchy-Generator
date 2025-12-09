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

import download_utils

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# --- Legacy ImageNet Logic (Bottom-Up from WNIDs) ---

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
        logger.error(f"Error finding synset for {wnid}: {e}")
        return None

def build_hierarchy_tree_legacy(wnids: List[str]) -> Dict[str, Any]:
    tree: Dict[str, Any] = {}
    for wnid in wnids:
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

def handle_imagenet_wnid(args) -> None:
    ensure_nltk_data()

    if not args.inputs:
        logger.info("No input provided. Using sample IDs.")
        wnids = ['n02084071', 'n02113799', 'n07753592'] # Simplified sample
    else:
        wnids = load_wnids(args.inputs)

    if not wnids:
        logger.warning("No WNIDs to process.")
        return

    logger.info(f"Processing {len(wnids)} IDs (Bottom-Up)...")
    hierarchy = build_hierarchy_tree_legacy(wnids)

    with open(args.output, 'w') as f:
        yaml.dump(hierarchy, f, sort_keys=False, default_flow_style=False, indent=4)
    logger.info(f"Saved to {args.output}")


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

# Re-implementing strictly following the user snippet logic but adding filter
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
        # If no children survived filter, and we are not a leaf in original graph,
        # we treat this node as a leaf candidate?
        # Or we prune it?
        # If I am 'carnivore', and I filtered all my children, do I keep 'carnivore'?
        # Probably not if we want a classification tree.
        # But if valid_wnids is None, we keep everything.
        if valid_wnids is None:
             return name # Treat as leaf if max depth reached or no children
        else:
             # Check if I am valid myself
             wnid = f"{synset.pos()}{synset.offset():08d}"
             if wnid in valid_wnids:
                 return name
             return None

    return child_nodes

def handle_imagenet_tree(args) -> None:
    ensure_nltk_data()
    
    valid_wnids = None
    if args.filter:
        logger.info("Ensuring ImageNet list is available...")
        list_path = download_utils.ensure_imagenet_list()
        valid_wnids = load_valid_wnids(list_path)
        logger.info(f"Loaded {len(valid_wnids)} valid WNIDs for filtering.")
    
    root_str = args.root
    try:
        root_synset = wn.synset(root_str)
    except Exception:
        logger.error(f"Could not find root synset: {root_str}")
        return

    logger.info(f"Building hierarchy from {root_str} (Top-Down, max_depth={args.depth})...")

    # We wrap the result in a dict with the root name
    root_name = root_synset.lemmas()[0].name().replace('_', ' ')
    content = build_hierarchy_snippet_style(root_synset, valid_wnids, max_depth=args.depth)

    if content:
        hierarchy = {root_name: content}
    else:
        hierarchy = {}
        logger.warning("Resulting hierarchy is empty (possibly due to aggressive filtering).")

    with open(args.output, 'w') as f:
        yaml.dump(hierarchy, f, sort_keys=False, default_flow_style=False, indent=4)
    logger.info(f"Saved to {args.output}")


# --- COCO Logic ---

def handle_coco(args) -> None:
    logger.info("Ensuring COCO data is available...")
    json_path = download_utils.ensure_coco_data()

    logger.info(f"Loading {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)

    logger.info("Processing COCO categories...")
    hierarchy = {}
    for cat in data['categories']:
        supercat = cat['supercategory']
        name = cat['name']

        if supercat not in hierarchy:
            hierarchy[supercat] = []

        hierarchy[supercat].append(name)

    with open(args.output, 'w') as f:
        yaml.dump(hierarchy, f, sort_keys=False)
    logger.info(f"Saved to {args.output}")


# --- Open Images Logic ---

def parse_openimages_node(node, id_to_name):
    label_id = node.get('LabelName')
    name = id_to_name.get(label_id, label_id)

    if 'Subcategories' in node:
        children = {}
        for sub in node['Subcategories']:
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

def handle_openimages(args) -> None:
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
    final_yaml = parse_openimages_node(data, id_to_name)

    with open(args.output, 'w') as f:
        yaml.dump(final_yaml, f, sort_keys=False)
    logger.info(f"Saved to {args.output}")


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
