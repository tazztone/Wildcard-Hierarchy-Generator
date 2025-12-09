# ImageNet Hierarchy Generator

A production-ready Python tool to generate a WordNet hierarchy YAML file from a list of ImageNet WordNet IDs (WNIDs).

## Features

- **Input Flexibility**: Accepts WNIDs directly via command line or through text files.
- **Robustness**: Includes logging, error handling, and type hinting.
- **Data Management**: Automatically downloads necessary NLTK WordNet data.
- **Customizable Output**: Specify the output file name and location.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-repo/wildcards-wordnet-yaml.git
    cd wildcards-wordnet-yaml
    ```

2.  **Install dependencies**:
    Ensure you have Python 3 installed. It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Basic Usage

Run the script with default sample data (if no input is provided):

```bash
python app.py
```

### Direct Input

Provide WNIDs directly as arguments:

```bash
python app.py n02084071 n02113799
```

### File Input

Provide a text file containing WNIDs (one per line):

```bash
python app.py inputs.txt
```

You can mix file paths and direct IDs:

```bash
python app.py inputs.txt n07747607
```

### Options

*   `-o, --output <file>`: Specify the output YAML file path (default: `imagenet_hierarchy.yaml`).
*   `-v, --verbose`: Enable verbose debug logging.
*   `-h, --help`: Show help message.

**Example with options:**

```bash
python app.py inputs.txt -o my_hierarchy.yaml -v
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

To run the test suite, first ensure `pytest` is installed (it is included in `requirements.txt`), then run:

```bash
pytest tests/
```

### Project Structure

*   `app.py`: Main application logic.
*   `tests/`: Unit tests.
*   `requirements.txt`: Python dependencies.
*   `AGENTS.md`: Guidelines for AI agents and contributors.
