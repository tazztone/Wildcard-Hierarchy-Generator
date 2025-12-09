import gradio as gr
import app_gradio

def test_gradio_demo_construction():
    assert isinstance(app_gradio.demo, gr.Blocks)
