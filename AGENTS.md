# AGENTS.md

## Context for AI Agents and Contributors

This file serves as the definitive guide for understanding the architecture, conventions, and operational procedures of the **Wildcard Hierarchy Generator** codebase.

### 1. Project Architecture

The project is structured around a core logic module (`app.py`), a GUI layer (`app_gradio.py`), and utility modules.

#### Core Components
-   **`app.py`**:
    -   **Role**: Entry point for CLI and container for core logic.
    -   **Key Functions**:
        -   `generate_imagenet_wnid_hierarchy()`: Bottom-up construction from WNID list.
        -   `generate_imagenet_tree_hierarchy()`: Top-down recursion from root synset.
        -   `generate_coco_hierarchy()`: Flat structure generation.
        -   `generate_openimages_hierarchy()`: Recursive parsing of JSON hierarchy.
        -   `convert_to_wildcard_format()`: Critical post-processing step. Converts raw nested dicts into the specific YAML format required (lists for leaves, etc.).
    -   **Data Flow**: `Load Data -> Process/Build Tree -> Post-process (Flatten/Format) -> Save YAML`.

-   **`app_gradio.py`**:
    -   **Role**: Gradio-based web interface.
    -   **Pattern**: Uses `gr.State` to track active tabs and parameters. Implements a dispatch pattern (`dispatch_preview`, `dispatch_save`) to handle events from multiple tabs using shared buttons.
    -   **Validation**: Real-time validation logic (`validate_inputs`).

-   **`download_utils.py`**:
    -   **Role**: Centralized handling of external data.
    -   **Behavior**: Checks for local existence in `downloads/` before fetching. Uses `tqdm` for progress bars.

### 2. Operational Guidelines

#### Hierarchy Generation Logic
-   **Recursion & Depth**: Most generators use recursion. The `max_depth` parameter is strict. When recursion hits `max_depth`, the `flatten_hierarchy_post_process` function (or equivalent logic inside the generator) collapses all valid descendants into a flat list.
-   **Leaf Nodes**: A "leaf" in the wildcard format is a string within a list. A non-leaf is a dictionary key.
-   **Filtering**: ImageNet filtering works by passing a set of valid WNIDs (`valid_wnids`). During traversal, if a node's ID is not in the set, it (and its children) may be pruned, or it might be included as a path to a valid descendant depending on the logic.
-   **Semantic Filtering**:
    -   **Primary Synset Filtering**: Enabled by default in recursive generation. Uses `get_primary_synset` to check if the current synset corresponds to the first (most common) meaning of its lemma name. If not, the node is pruned. This prevents "old man" (slang) from appearing under "communication".
    -   **Blacklisting**: Optional. Prunes predefined abstract categories (e.g., `communication`, `entity`) to remove high-level noise.
    -   **Hypernym Depth**: Optional for Bottom-Up generation. Limits the height of the hypernym path from the leaf, creating a forest of categories instead of a single deep tree rooted at `entity`.

#### File I/O & Encoding
-   **Strict UTF-8**: Always use `encoding='utf-8'` for `open()`. This is critical for Windows compatibility.
-   **Downloads Directory**: All external data must go to `downloads/`. Do not pollute the root directory.

#### Testing Strategy
-   **Framework**: `pytest`.
-   **Mocking**:
    -   **Network**: Never make real network calls in tests. Mock `urllib.request` or the `download_utils` functions.
    -   **NLTK**: NLTK lazy loaders are tricky. Mock `app.wn` or `nltk.corpus.wordnet` directly.
    -   **UI**: UI logic is tested in `tests/test_gradio.py` by mocking the backend `generate_*` functions and testing the `on_preview`/`on_save` wrappers.

### 3. Coding Standards

-   **Style**: PEP 8.
-   **Type Hints**: Mandatory for all function signatures.
-   **Docstrings**: Required. Explanation of arguments and return values.
-   **Logging**: Use `logging.getLogger(__name__)`. Do not use `print()` in library code.

### 4. Memory & Optimization

-   **Large Datasets**: ImageNet-21K and Open Images are large.
    -   Avoid loading full datasets into memory if possible.
    -   Use generators/iterators where applicable (though current implementation mostly builds full dicts).
    -   The GUI uses truncation (`format_yaml_preview`) to prevent browser crashes when displaying large outputs.

### 5. Known Issues / Gotchas

-   **NLTK Data**: The first run requires `wordnet` corpus. `ensure_nltk_data()` handles this, but it can fail if no internet.
-   **Gradio State**: `gr.State` initialization must match the UI default values. Mismatches can cause "NoneType" errors in event handlers.
-   **Wildcard Format**: The format is sensitive. `{"key": ["item"]}` is different from `{"key": {"item": {}}}`. Use `convert_to_wildcard_format` to standardize.

### 6. Adding New Datasets

1.  Add a loader in `download_utils.py`.
2.  Implement a `generate_<dataset>_hierarchy` function in `app.py`.
3.  Add a CLI parser in `app.py`.
4.  Add a new Tab in `app_gradio.py` and update the dispatch functions.
5.  Add tests.

---
**Note**: This file should be updated whenever significant architectural changes are made.
