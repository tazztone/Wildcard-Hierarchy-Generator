import gradio as gr
import app_gradio
from unittest.mock import patch, MagicMock

def test_gradio_demo_construction():
    assert isinstance(app_gradio.demo, gr.Blocks)

def test_preview_wnid():
    with patch('app.generate_imagenet_wnid_hierarchy') as mock_gen:
        mock_gen.return_value = {'dog': {}}
        prev, stats = app_gradio.preview_imagenet_wnid("n12345678")
        assert "dog" in prev
        assert "Total items: 1" in stats

def test_preview_21k():
    with patch('app.download_utils.ensure_imagenet21k_data') as mock_ensure:
        with patch('app.load_wnids') as mock_load:
            with patch('app.generate_imagenet_wnid_hierarchy') as mock_gen:
                mock_ensure.return_value = ('ids.txt', 'lemmas.txt')
                mock_load.return_value = ['n00000001'] * 60 # 60 items
                mock_gen.return_value = {'test': {}}

                prev, stats = app_gradio.preview_21k()

                assert "Preview showing first 50" in stats
                assert "Total items: 60" in stats
                mock_gen.assert_called()
                # Should be called with subset
                args, _ = mock_gen.call_args
                assert len(args[0]) == 50

def test_preview_tree():
    with patch('app.generate_imagenet_tree_hierarchy') as mock_gen:
        mock_gen.return_value = {'root': {'child': 'child'}}
        prev, stats = app_gradio.preview_tree('root', 1, False)
        assert "root" in prev
        assert "Generated" in stats

def test_preview_coco():
    with patch('app.generate_coco_hierarchy') as mock_gen:
        mock_gen.return_value = {'super': ['cat1', 'cat2']}
        prev, stats = app_gradio.preview_coco()
        assert "super" in prev
        assert "Found 1 supercategories" in stats
