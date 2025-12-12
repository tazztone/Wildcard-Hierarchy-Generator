# Wildcard Hierarchy Generator

A powerful Python tool designed to generate WordNet-style hierarchy YAML files from major computer vision datasets: **ImageNet**, **COCO**, and **Open Images**.

This tool is particularly useful for creating wildcard files for dynamic prompt generation in AI art tools (like Stable Diffusion web UI) or for organizing dataset labels hierarchically.

## 🌟 Features

-   **ImageNet Support**:
    -   **Tree Mode**: Generate a top-down hierarchy starting from any root synset (e.g., `entity.n.01`), with optional filtering for **ImageNet-1k** (1000 classes) or **ImageNet-21k**.
    -   **Custom List Mode**: Build a bottom-up hierarchy from a custom list of WordNet IDs (WNIDs).
-   **COCO Support**: Generate a clean hierarchy from the COCO 80-category object detection dataset.
-   **Open Images Support**: Generate a deep hierarchy from the Open Images V7 dataset (600+ classes).
-   **Dual Interface**:
    -   **CLI**: Robust command-line interface for automation and scripting.
    -   **GUI**: User-friendly Web UI based on Gradio for interactive exploration and preview.
-   **Wildcard Optimized**: The output YAML is formatted specifically for wildcard compatibility (lists for leaf nodes, nested dicts for structure).
-   **Smart Caching**: Automatically downloads and caches necessary metadata (WordNet, class lists) to a `downloads/` directory.

## 🚀 Installation

### Prerequisites
-   Python 3.8 or higher
-   Git

### Steps

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-repo/wildcards-wordnet-yaml.git
    cd wildcards-wordnet-yaml
    ```

2.  **Create a Virtual Environment (Recommended)**:
    ```bash
    # Linux/macOS
    python3 -m venv venv
    source venv/bin/activate

    # Windows
    python -m venv venv
    venv\Scripts\activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 🖥️ Usage

You can use the tool via the Command Line Interface (CLI) or the Graphical User Interface (GUI).

### Graphical User Interface (GUI)

Launch the interactive web interface:

```bash
python app_gradio.py
```

Open your browser at `http://localhost:7860`. The GUI allows you to:
-   Select dataset modes via tabs.
-   Visualize the hierarchy structure before saving.
-   Upload WNID lists for custom generation.
-   Configure depth and filtering options with real-time validation.

### Command Line Interface (CLI)

The `app.py` script serves as the main entry point.

#### 1. ImageNet (Tree Mode)
Generate a hierarchy top-down from a root node.

```bash
# Generate full WordNet hierarchy for animals, max depth 5
python app.py imagenet-tree --root animal.n.01 --depth 5 --output animals.yaml

# Generate hierarchy for ImageNet-1k classes only
python app.py imagenet-tree --filter 1k --output imagenet1k.yaml

# Generate hierarchy for ImageNet-21k classes
python app.py imagenet-tree --filter 21k --output imagenet21k.yaml
```

**Arguments:**
-   `--root`: The root synset to start from (default: `entity.n.01`).
-   `--depth`: Maximum recursion depth before flattening.
-   `--filter`: Filter classes: `none` (default), `1k`, or `21k`.
-   `-o, --output`: Output YAML file path.

#### 2. ImageNet (Custom List Mode)
Generate a hierarchy from a specific list of WordNet IDs (WNIDs).

```bash
# From a file containing WNIDs
python app.py imagenet-wnid input_wnids.txt -o my_custom_hierarchy.yaml

# From command line arguments
python app.py imagenet-wnid n02084071 n02121808 -o dogs_cats.yaml
```

**Arguments:**
-   `inputs`: List of WNIDs or path to a text file containing them.
-   `--depth`: Maximum depth for the resulting structure.
-   `-o, --output`: Output YAML file path.

#### 3. COCO
Generate the hierarchy for the COCO dataset.

```bash
python app.py coco --output output/coco.yaml
```
*Note: This will download `annotations_trainval2017.zip` if `coco_categories.json` is not present in `downloads/`.*

#### 4. Open Images
Generate the hierarchy for Open Images V7.

```bash
python app.py openimages --output output/openimages.yaml
```

### Automation Scripts

Helper scripts are provided in the `scripts/` directory to generate standard hierarchies in bulk.

**Linux/macOS:**
```bash
bash scripts/generate_all.sh
```

**Windows:**
```batch
scripts\generate_all.bat
```

## 📂 Output Format

The tool generates YAML files optimized for wildcard usage, specifically targeting compatibility with **[ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-extension-tutorials/blob/Main/ComfyUI-Impact-Pack/tutorial/ImpactWildcard.md)**.

**Key Characteristics:**
-   **ImpactWildcard Compatible**: The structure corresponds directly to the directory/file path format expected by the Impact Pack's wildcard system (e.g., `__category/subcategory/item__`).
-   **Nested Structure**: Represents the hierarchy tree as nested dictionaries.
-   **Leaf Nodes**: Represented as lists of strings (e.g., `[dog, puppy]`).
-   **Mixed Nodes**: If a node has both children and is a valid category itself, it uses a self-reference pattern (though primarily the structure attempts to push items to leaves).
-   **Flattening**: When `max_depth` is reached, all descendants are flattened into a single list under that node.

**Example:**
```yaml
entity:
  physical_entity:
    object:
      living_thing:
        organism:
          animal:
            canine:
              - dog
              - wolf
            feline:
              - cat
              - lion
```

## 🛠️ Development

### Project Structure
-   `app.py`: Core CLI logic and hierarchy generation algorithms.
-   `app_gradio.py`: Gradio-based GUI implementation.
-   `download_utils.py`: Helper functions for downloading and caching external datasets.
-   `scripts/`: Automation scripts.
-   `downloads/`: Default directory for cached data (created on first run).
-   `output/`: Default directory for generated files.
-   `tests/`: Unit tests (pytest).

### Running Tests
Run the test suite to ensure everything is working correctly:

```bash
pytest tests/
```

## ❓ Troubleshooting

**Q: "Resource 'wordnet' not found" error?**
A: The tool attempts to download NLTK data automatically. If this fails (e.g., due to firewall), run python and execute:
```python
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
```

**Q: Slow downloads?**
A: The first run for COCO or Open Images may take time as it downloads dataset metadata. Subsequent runs will use the cached files in `downloads/`.

**Q: Encoding errors on Windows?**
A: Ensure your console supports UTF-8. The tool explicitly uses `utf-8` for file operations.

## 🤝 Contributing

Contributions are welcome! Please check `AGENTS.md` for coding standards and architectural guidelines.

1.  Fork the repo.
2.  Create a feature branch.
3.  Commit your changes.
4.  Push to the branch.
5.  Create a Pull Request.

## 📄 License

[License Name]
