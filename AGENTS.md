# AGENTS.md

This file provides context and guidelines for AI agents and human contributors working on this codebase.

## Project Overview

This project is a utility to generate a WordNet hierarchy from ImageNet WNIDs. It uses the NLTK library to traverse the WordNet graph and build a nested dictionary structure, which is then exported as YAML.

## Coding Standards

*   **Python Version**: Python 3.8+.
*   **Style**: Follow PEP 8 guidelines.
*   **Type Hinting**: All functions must have type hints for arguments and return values.
*   **Docstrings**: All functions and modules must have descriptive docstrings (Google style or standard Sphinx style).
*   **Logging**: Use the `logging` module instead of `print` for application messages. `print` is acceptable only for CLI output specifically requested by flags (e.g., debug dumps) or user interaction if interactive features are added.

## Testing

*   **Framework**: `pytest`.
*   **Coverage**: Ensure new features are covered by unit tests.
*   **Mocking**: Use `unittest.mock` to avoid external dependencies (like network calls to download NLTK data) during tests. Note that NLTK's lazy loading can sometimes make patching tricky; patching the object instance in the module (e.g., `app.wn`) is often more reliable than patching the import source.

## File Structure

*   `app.py`: The entry point and main logic. Keep this file focused. If logic grows significantly, refactor into a `src/` package.
*   `tests/`: Contains all test files. Naming convention: `test_<module_name>.py`.
*   `requirements.txt`: Pin dependencies here.

## Common Tasks

*   **Adding new features**: Ensure `app.py` handles arguments via `argparse`.
*   **Updating dependencies**: Run `pip freeze` and update `requirements.txt` carefully, only including direct dependencies if possible, or all if locking is desired.

## Notes for Agents

*   When modifying `app.py`, check if the NLTK data handling needs adjustment.
*   The `ensure_nltk_data` function should run before any logic that requires WordNet.
*   Be aware that `wnid` usually starts with 'n' followed by 8 digits (noun offset).
