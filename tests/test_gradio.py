import gradio as gr
import app_gradio
from unittest.mock import patch, MagicMock

def test_gradio_demo_construction():
    demo = app_gradio.create_ui()
    assert isinstance(demo, gr.Blocks)

def test_on_preview_wnid():
    # mode, strategy, wnid_text, root_synset, filter_mode, max_depth
    with patch('app_gradio.generate_hierarchy') as mock_gen:
        mock_gen.return_value = {'dog': {}}
        prev, stats = app_gradio.on_preview("ImageNet", "Custom List", "n12345678", "entity.n.01", "All WordNet", 3)

        # Verify call arguments
        mock_gen.assert_called_with("ImageNet", "Custom List", "n12345678", "entity.n.01", "All WordNet", 3)

        assert "dog" in prev
        assert "Preview generated successfully" in stats

def test_on_preview_tree():
    with patch('app_gradio.generate_hierarchy') as mock_gen:
        mock_gen.return_value = {'root': {'child': 'child'}}
        prev, stats = app_gradio.on_preview("ImageNet", "Recursive (from Root)", "", "root", "All WordNet", 3)

        mock_gen.assert_called_with("ImageNet", "Recursive (from Root)", "", "root", "All WordNet", 3)

        assert "root" in prev
        assert "Preview generated successfully" in stats

def test_on_preview_coco():
    with patch('app_gradio.generate_hierarchy') as mock_gen:
        mock_gen.return_value = {'super': ['cat1', 'cat2']}
        # Strategy for COCO is ignored/passed as is
        prev, stats = app_gradio.on_preview("COCO", "Default", "", "", "All WordNet", 1)
        assert "super" in prev
        assert "Preview generated successfully" in stats

def test_on_preview_validation_empty_wnid():
    prev, stats = app_gradio.on_preview("ImageNet", "Custom List", "", "entity.n.01", "All WordNet", 3)
    assert "Please enter at least one WNID" in prev
    assert "Please enter at least one WNID" in stats

def test_on_preview_defaults_empty_root():
    # If root is empty, it should default to entity.n.01 and succeed
    with patch('app_gradio.generate_hierarchy') as mock_gen:
        mock_gen.return_value = {}
        prev, stats = app_gradio.on_preview("ImageNet", "Recursive (from Root)", "", "", "All WordNet", 3)

        # Check that it called with entity.n.01
        mock_gen.assert_called_with("ImageNet", "Recursive (from Root)", "", "entity.n.01", "All WordNet", 3)
        assert "Preview generated successfully" in stats
