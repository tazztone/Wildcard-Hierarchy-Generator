# Wildcard Hierarchy Generator

A Python tool to generate WordNet-style hierarchy YAML files from various datasets (ImageNet, COCO, Open Images).

## Features

- **ImageNet**: Generate hierarchies from ImageNet WNIDs (legacy) or the full ImageNet tree filtered to the standard 1000 classes.
- **COCO**: Generate hierarchy from COCO categories (supports local optimization to avoid large downloads).
- **Open Images**: Generate hierarchy from Open Images V5 data.
- **Automation**: Includes scripts to generate all hierarchies in one go.

## Datasets

Detailed information about the datasets supported by this tool:

### ImageNet
- **Website**: [http://www.image-net.org/](http://www.image-net.org/)
- **Overview**: ImageNet is an image database organized according to the WordNet hierarchy (currently only the nouns), in which each node of the hierarchy is depicted by hundreds and thousands of images. The project has been instrumental in advancing computer vision and deep learning research.
- **Usage**: This tool uses NLTK's interface to WordNet to generate hierarchies based on ImageNet's structure (WordNet 3.0).

### COCO (Common Objects in Context)
- **Website**: [https://cocodataset.org/](https://cocodataset.org/)
- **Overview**: COCO is a large-scale object detection, segmentation, and captioning dataset. It features 80 object categories (like person, bicycle, car) and is widely used for benchmarking object detection models.
- **Usage**: The hierarchy is flat (Categories -> Supercategories).

### Open Images
- **Website**: [https://storage.googleapis.com/openimages/web/index.html](https://storage.googleapis.com/openimages/web/index.html)
- **Overview**: Open Images is a dataset of ~9 million images annotated with image-level labels, object bounding boxes, object segmentation masks, visual relationships, and localized narratives. It contains a much larger number of categories than COCO or ImageNet 1k.
- **Usage**: The hierarchy is derived from the `bbox_labels_600_hierarchy.json` provided by Open Images.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-repo/wildcards-wordnet-yaml.git
    cd wildcards-wordnet-yaml
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Quick Start (Generate All)

To generate all hierarchies (ImageNet, COCO, Open Images) into the `output/` directory:

**Linux/macOS:**
```bash
bash scripts/generate_all.sh
```

**Windows:**
```batch
scripts\generate_all.bat
```

The generated files will be:
- `output/imagenet.yaml`
- `output/coco.yaml`
- `output/openimages.yaml`

### Manual Usage

You can also run the tool manually via `app.py`.

#### ImageNet (Top-Down Tree)
Generates a hierarchy starting from a root node, optionally filtered to the ImageNet 1k classes.

```bash
python app.py imagenet-tree --root entity.n.01 --depth 25 --filter --output output/imagenet.yaml
```
*   `--root`: The root synset (default: `animal.n.01`).
*   `--depth`: Max recursion depth.
*   `--filter`: Filter to include only paths leading to the standard ImageNet 1k classes.

#### COCO
Generates the COCO 80-category hierarchy.

```bash
python app.py coco --output output/coco.yaml
```
*   **Optimization**: If `coco_categories.json` is present in the root directory, the tool uses it instead of downloading the large (200MB+) annotations zip file.

#### Open Images
Generates the Open Images V5 hierarchy.

```bash
python app.py openimages --output output/openimages.yaml
```

#### Legacy ImageNet (Bottom-Up)
Generates a hierarchy from a list of WNIDs provided in a file or arguments.

```bash
python app.py imagenet-wnid inputs.txt -o my_hierarchy.yaml
```

## Output Format

The output is a YAML file representing the hierarchy tree:

```yaml
entity:
    physical_entity:
        object:
            living_thing:
                organism:
                    animal:
                        ...
                            dog: {}
```

## Development

### Running Tests

To run the test suite:

```bash
pytest tests/
```

### Project Structure

*   `app.py`: Main application logic.
*   `download_utils.py`: Utilities for downloading external data.
*   `scripts/`: Helper scripts for generation.
*   `coco_categories.json`: Lightweight COCO category list (to avoid large downloads).
*   `tests/`: Unit tests.
