import gradio as gr
import app_gradio
from unittest.mock import patch, MagicMock
import app

def test_gradio_demo_construction():
    demo = app_gradio.create_ui()
    assert isinstance(demo, gr.Blocks)

def test_on_preview_wnid():
    with patch('app_gradio.generate_hierarchy') as mock_gen:
        mock_gen.return_value = {'dog': {}}
        prev, stats = app_gradio.on_preview("ImageNet", "Custom List", "n12345678", "entity.n.01", "All WordNet", 3, True, False, 0)
        mock_gen.assert_called_with("ImageNet", "Custom List", "n12345678", "entity.n.01", "All WordNet", 3, True, False, 0)
        assert "dog" in prev
        assert "Preview generated successfully" in stats

def test_on_preview_tree():
    with patch('app_gradio.generate_hierarchy') as mock_gen:
        mock_gen.return_value = {'root': {'child': 'child'}}
        prev, stats = app_gradio.on_preview("ImageNet", "Recursive (from Root)", "", "root", "All WordNet", 3, True, False, 0)
        mock_gen.assert_called_with("ImageNet", "Recursive (from Root)", "", "root", "All WordNet", 3, True, False, 0)
        assert "root" in prev
        assert "Preview generated successfully" in stats

def test_on_preview_coco():
    with patch('app_gradio.generate_hierarchy') as mock_gen:
        mock_gen.return_value = {'super': ['cat1', 'cat2']}
        prev, stats = app_gradio.on_preview("COCO", "Default", "", "", "All WordNet", 1, True, False, 0)
        assert "super" in prev
        assert "Preview generated successfully" in stats

def test_on_preview_validation_empty_wnid():
    prev, stats = app_gradio.on_preview("ImageNet", "Custom List", "", "entity.n.01", "All WordNet", 3, True, False, 0)
    assert "Please enter at least one WNID" in prev
    assert "Please enter at least one WNID" in stats

def test_on_preview_defaults_empty_root():
    with patch('app_gradio.generate_hierarchy') as mock_gen:
        mock_gen.return_value = {}
        prev, stats = app_gradio.on_preview("ImageNet", "Recursive (from Root)", "", "", "All WordNet", 3, True, False, 0)
        mock_gen.assert_called_with("ImageNet", "Recursive (from Root)", "", "entity.n.01", "All WordNet", 3, True, False, 0)
        assert "Preview generated successfully" in stats

def test_on_tab_change():
    evt = MagicMock()
    evt.index = 1
    mode, filename = app_gradio.on_tab_change(evt)
    assert mode == "COCO"
    assert filename == "wildcards_coco.yaml"

    evt.index = 2
    mode, filename = app_gradio.on_tab_change(evt)
    assert mode == "Open Images"
    assert filename == "wildcards_openimages.yaml"

def test_dispatch_save_download():
    # Test that dispatch_save returns correct tuple for download button
    with patch('app_gradio.generate_hierarchy') as mock_gen, \
         patch('app.save_hierarchy') as mock_save:
        mock_gen.return_value = {'dog': {}}

        # Note: We need to ensure dispatch_save is available at module level now
        res = app_gradio.dispatch_save(
            "ImageNet",
            "Custom List", "entity.n.01", "All WordNet", "n12345", True, False, 0,
            "Default", "entity.n.01", "All WordNet", "",
            "Default", "entity.n.01", "All WordNet", "",
            3, "out.yaml"
        )

        assert isinstance(res, tuple)
        assert len(res) == 2
        status, update = res

        assert "Successfully saved" in status
        # update is a dict in recent Gradio versions for updates
        assert update['visible'] is True
        assert update['value'] == "out.yaml"
