import gradio as gr
import app_gradio
from unittest.mock import patch, MagicMock

def test_gradio_demo_construction():
    demo = app_gradio.create_ui()
    assert isinstance(demo, gr.Blocks)

def test_on_preview_wnid():
    # mode, wnid_text, root_synset, filter_mode, max_depth
    with patch('app_gradio.generate_hierarchy') as mock_gen:
        mock_gen.return_value = {'dog': {}}
        prev, stats = app_gradio.on_preview("ImageNet (Custom List)", "n12345678", "root", "All WordNet", 3)
        assert "dog" in prev
        assert "Preview generated successfully" in stats

def test_on_preview_tree():
    with patch('app_gradio.generate_hierarchy') as mock_gen:
        mock_gen.return_value = {'root': {'child': 'child'}}
        prev, stats = app_gradio.on_preview("ImageNet (Tree)", "", "root", "All WordNet", 3)
        assert "root" in prev
        assert "Preview generated successfully" in stats

def test_on_preview_coco():
    with patch('app_gradio.generate_hierarchy') as mock_gen:
        mock_gen.return_value = {'super': ['cat1', 'cat2']}
        prev, stats = app_gradio.on_preview("COCO", "", "", "All WordNet", 1)
        assert "super" in prev
        assert "Preview generated successfully" in stats

def test_on_preview_validation_empty_wnid():
    prev, stats = app_gradio.on_preview("ImageNet (Custom List)", "", "root", "All WordNet", 3)
    assert "Please enter at least one WNID" in prev
    assert "Please enter at least one WNID" in stats
