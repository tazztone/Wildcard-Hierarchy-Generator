#!/usr/bin/env python3
"""
ImageNet Hierarchy Generator

This script generates a WordNet hierarchy YAML from a list of ImageNet WNIDs.
It accepts WNIDs directly via command-line arguments or through text files.
"""

import argparse
import logging
import os
import sys
from typing import List, Dict, Optional, Any

import nltk
from nltk.corpus import wordnet as wn
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

def ensure_nltk_data() -> None:
    """
    Ensures that the necessary NLTK WordNet data is available.
    Downloads it if not present.
    """
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
    """
    Converts an ImageNet WNID (e.g., 'n02084071') to an NLTK Synset object.

    ImageNet IDs are formatted as: <pos><offset>, e.g., 'n' + '02084071'.

    Args:
        wnid: The WordNet ID string.

    Returns:
        The NLTK Synset object or None if not found/error.
    """
    try:
        if len(wnid) < 2:
            logger.warning(f"Invalid WNID format: {wnid}")
            return None

        pos = wnid[0] # 'n' for noun
        offset_str = wnid[1:]

        if not offset_str.isdigit():
             logger.warning(f"Invalid WNID offset (must be digits): {wnid}")
             return None

        offset = int(offset_str) # 02084071

        # wordnet.synset_from_pos_and_offset is the bridge between ID and hierarchy
        return wn.synset_from_pos_and_offset(pos, offset)
    except Exception as e:
        logger.error(f"Error finding synset for {wnid}: {e}")
        return None

def build_hierarchy_tree(wnids: List[str]) -> Dict[str, Any]:
    """
    Takes a list of WNIDs and builds a nested dictionary representing the 
    WordNet hierarchy.

    Args:
        wnids: A list of WordNet IDs.

    Returns:
        A nested dictionary representing the hierarchy.
    """
    tree: Dict[str, Any] = {}

    for wnid in wnids:
        synset = get_synset_from_wnid(wnid)
        if not synset:
            continue

        # Get paths to the root 'entity'. 
        # Note: WordNet is a DAG (Directed Acyclic Graph), so a synset can have 
        # multiple parent paths. We usually take the first one for a clean tree structure.
        paths = synset.hypernym_paths()
        
        if not paths:
            logger.warning(f"No hypernym paths found for {wnid}")
            continue

        # We take the path that is likely the most descriptive (often the longest)
        # or simply the first one provided by NLTK.
        primary_path = paths[0] 
        
        # The path includes the root down to the leaf.
        # We need to merge this path into our main 'tree' dictionary.
        current_level = tree
        for node in primary_path:
            # We use a string representation "name (id)" for the key
            # wn.synset identifier looks like 'dog.n.01'
            node_name = node.name().split('.')[0]
            
            # Format: "name (offset)" to ensure uniqueness if names collide
            # converting offset back to string id: n + 8 digit padded
            # Note: We are not using the offset in the key name in the user's example,
            # just "node_name". If collision handling is needed, we might need to adjust.
            # But the logic below follows the user's snippet: key = f"{node_name}"
            
            # Create a descriptive key
            key = f"{node_name}"
            
            # Initialize this key if not present
            if key not in current_level:
                current_level[key] = {}
            
            # Move down into this node
            current_level = current_level[key]
            
        # At the end of the path (the specific breed/object), we can leave an empty dict
        # or mark it as a leaf. The loop naturally leaves it as a dict key pointing to {}.

    return tree

def load_wnids(inputs: List[str]) -> List[str]:
    """
    Loads WNIDs from a list of inputs which can be direct IDs or file paths.

    Args:
        inputs: List of strings (IDs or file paths).

    Returns:
        List of unique WNIDs.
    """
    wnids_to_process = []

    for input_str in inputs:
        if os.path.isfile(input_str):
            # It's a file, read IDs from it
            try:
                with open(input_str, 'r') as f:
                    lines = [line.strip() for line in f if line.strip()]
                    logger.info(f"Loaded {len(lines)} IDs from file: {input_str}")
                    wnids_to_process.extend(lines)
            except Exception as e:
                logger.error(f"Error reading file {input_str}: {e}")
        else:
            # It's a direct WNID
            wnids_to_process.append(input_str)

    # Remove duplicates
    return list(dict.fromkeys(wnids_to_process))

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a WordNet hierarchy YAML from ImageNet WNIDs.")
    parser.add_argument('inputs', nargs='*', help="List of WNIDs (e.g., n02084071) or paths to text files containing WNIDs.")
    parser.add_argument('-o', '--output', default='imagenet_hierarchy.yaml', help="Output YAML file path (default: imagenet_hierarchy.yaml)")
    parser.add_argument('-v', '--verbose', action='store_true', help="Enable verbose debug logging")

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    ensure_nltk_data()

    wnids_to_process = []

    # If no inputs provided, fall back to a hardcoded sample for demonstration
    if not args.inputs:
        logger.info("No input provided. Using sample ImageNet IDs for demonstration.")
        wnids_to_process = [
            'n02084071', # Dog (generic)
            'n02113799', # Siberian husky
            'n02110806', # Basenji
            'n02124075', # Egyptian cat
            'n03595614', # Jersey (cattle)
            'n04467665', # Trailer truck
            'n04285008', # Sports car
            'n07747607', # Orange (fruit)
            'n07753592', # Banana
        ]
    else:
        wnids_to_process = load_wnids(args.inputs)
    
    if not wnids_to_process:
        logger.warning("No valid WNIDs found to process.")
        return

    logger.info(f"Processing {len(wnids_to_process)} ImageNet IDs...")
    
    hierarchy = build_hierarchy_tree(wnids_to_process)
    
    # Dump to YAML
    # We use a custom sorting to keep it somewhat tidy
    try:
        yaml_output = yaml.dump(hierarchy, sort_keys=False, default_flow_style=False, indent=4)

        if args.verbose:
             print("\n--- Generated YAML Hierarchy ---\n")
             print(yaml_output)

        # Save to file
        with open(args.output, 'w') as f:
            f.write(yaml_output)
        logger.info(f"Saved hierarchy to '{args.output}'")
    except Exception as e:
        logger.error(f"Failed to save YAML output: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
