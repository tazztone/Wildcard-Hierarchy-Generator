import nltk
from nltk.corpus import wordnet as wn
import yaml
import sys

# Ensure WordNet data is downloaded
try:
    wn.ensure_loaded()
except LookupError:
    print("Downloading WordNet data...")
    nltk.download('wordnet')
    nltk.download('omw-1.4')

def get_synset_from_wnid(wnid):
    """
    Converts an ImageNet WNID (e.g., 'n02084071') to an NLTK Synset object.
    ImageNet IDs are formatted as: <pos><offset>, e.g., 'n' + '02084071'
    """
    try:
        pos = wnid[0] # 'n' for noun
        offset = int(wnid[1:]) # 02084071
        # wordnet.synset_from_pos_and_offset is the bridge between ID and hierarchy
        return wn.synset_from_pos_and_offset(pos, offset)
    except Exception as e:
        print(f"Error finding synset for {wnid}: {e}")
        return None

def build_hierarchy_tree(wnids):
    """
    Takes a list of WNIDs and builds a nested dictionary representing the 
    WordNet hierarchy.
    """
    tree = {}

    for wnid in wnids:
        synset = get_synset_from_wnid(wnid)
        if not synset:
            continue

        # Get paths to the root 'entity'. 
        # Note: WordNet is a DAG (Directed Acyclic Graph), so a synset can have 
        # multiple parent paths. We usually take the first one for a clean tree structure.
        paths = synset.hypernym_paths()
        
        if not paths:
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
            node_id = f"n{str(node.offset()).zfill(8)}"
            
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

def main():
    # --- Configuration ---
    # A sample list of ImageNet WNIDs (WordNet IDs).
    # In a real scenario, you would load these from the ImageNet mapping file (map_clsloc.txt).
    sample_wnids = [
        'n02084071', # Dog (generic) -> actually specific synset might vary, let's use:
        'n02113799', # Siberian husky
        'n02110806', # Basenji
        'n02124075', # Egyptian cat
        'n03595614', # Jersey (cattle)
        'n04467665', # Trailer truck
        'n04285008', # Sports car
        'n07747607', # Orange (fruit)
        'n07753592', # Banana
    ]
    
    print(f"Processing {len(sample_wnids)} ImageNet IDs...")
    
    hierarchy = build_hierarchy_tree(sample_wnids)
    
    # Dump to YAML
    # We use a custom sorting to keep it somewhat tidy
    yaml_output = yaml.dump(hierarchy, sort_keys=False, default_flow_style=False, indent=4)
    
    print("\n--- Generated YAML Hierarchy ---\n")
    print(yaml_output)
    
    # Save to file
    with open('imagenet_hierarchy.yaml', 'w') as f:
        f.write(yaml_output)
    print("Saved to 'imagenet_hierarchy.yaml'")

if __name__ == "__main__":
    main()
