#!/usr/bin/env python3
"""
Gradio GUI for Hierarchy Generator
"""

import gradio as gr
import app
import yaml
import os
import logging

# Set up logging to capture output if needed, but app.py logs to stderr/stdout
logger = logging.getLogger("gui")

def format_yaml(data):
    return yaml.dump(data, sort_keys=False, default_flow_style=False, indent=4)

def safe_execution(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        return f"Error: {str(e)}"

# --- Tab Functions ---

def preview_imagenet_wnid(wnid_text):
    if not wnid_text.strip():
        return "Please enter at least one WNID."
    wnids = [line.strip() for line in wnid_text.split('\n') if line.strip()]

    def _run():
        h = app.generate_imagenet_wnid_hierarchy(wnids)
        return format_yaml(h)

    return safe_execution(_run)

def save_imagenet_wnid(wnid_text, output_path):
    if not wnid_text.strip():
        return "Please enter at least one WNID."
    wnids = [line.strip() for line in wnid_text.split('\n') if line.strip()]

    def _run():
        h = app.generate_imagenet_wnid_hierarchy(wnids)
        app.save_hierarchy(h, output_path)
        return f"Successfully saved to {output_path}"

    return safe_execution(_run)

def load_file_content(file_path):
    if not file_path:
        return ""
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"

def preview_tree(root, depth, filter_en):
    def _run():
        h = app.generate_imagenet_tree_hierarchy(root, int(depth), filter_en)
        return format_yaml(h)
    return safe_execution(_run)

def save_tree(root, depth, filter_en, path):
    def _run():
        h = app.generate_imagenet_tree_hierarchy(root, int(depth), filter_en)
        app.save_hierarchy(h, path)
        return f"Successfully saved to {path}"
    return safe_execution(_run)

def preview_coco():
    def _run():
        h = app.generate_coco_hierarchy()
        return format_yaml(h)
    return safe_execution(_run)

def save_coco(path):
    def _run():
        h = app.generate_coco_hierarchy()
        app.save_hierarchy(h, path)
        return f"Successfully saved to {path}"
    return safe_execution(_run)

def preview_oi():
    def _run():
        h = app.generate_openimages_hierarchy()
        return format_yaml(h)
    return safe_execution(_run)

def save_oi(path):
    def _run():
        h = app.generate_openimages_hierarchy()
        app.save_hierarchy(h, path)
        return f"Successfully saved to {path}"
    return safe_execution(_run)

# --- GUI Construction ---

with gr.Blocks(title="Hierarchy Generator") as demo:
    gr.Markdown("# Wildcard Hierarchy Generator")
    gr.Markdown("Generate YAML hierarchies for ImageNet, COCO, and Open Images datasets. This tool creates nested dictionary structures representing the class hierarchies.")

    with gr.Tabs():
        # --- ImageNet WNID Tab ---
        with gr.TabItem("ImageNet WNID"):
            gr.Markdown("### Bottom-Up Hierarchy from WNIDs")
            gr.Markdown("Provide a list of ImageNet WordNet IDs (WNIDs) to generate a hierarchy containing those classes and their ancestors. "
                        "This is useful if you have a specific subset of ImageNet classes.")

            with gr.Row():
                with gr.Column(scale=2):
                    wnid_input = gr.Textbox(
                        lines=12,
                        label="WNIDs (one per line)",
                        placeholder="n02084071\nn02113799",
                        info="Enter WNIDs here manually or upload a file."
                    )
                with gr.Column(scale=1):
                    file_upload = gr.File(
                        label="Upload WNID List (Text/File)",
                        file_count="single",
                        type="filepath"
                    )
                    gr.Markdown("*Uploading a file will replace the text in the box.*")

            output_path_wnid = gr.Textbox(value="imagenet_hierarchy.yaml", label="Output Filename")

            with gr.Row():
                btn_preview_wnid = gr.Button("Preview YAML", variant="secondary")
                btn_save_wnid = gr.Button("Generate & Save File", variant="primary")

            out_preview_wnid = gr.Code(language="yaml", label="Preview")
            out_status_wnid = gr.Textbox(label="Status", interactive=False)

            # Event handlers
            file_upload.change(load_file_content, inputs=[file_upload], outputs=[wnid_input])
            btn_preview_wnid.click(preview_imagenet_wnid, inputs=[wnid_input], outputs=[out_preview_wnid])
            btn_save_wnid.click(save_imagenet_wnid, inputs=[wnid_input, output_path_wnid], outputs=[out_status_wnid])

        # --- ImageNet Tree Tab ---
        with gr.TabItem("ImageNet Tree"):
            gr.Markdown("### Top-Down Recursive Hierarchy")
            gr.Markdown("Generate a hierarchy by starting from a root Synset and recursively finding children up to a certain depth. "
                        "This is useful for exploring subtrees of WordNet.")

            with gr.Row():
                root_input = gr.Textbox(value="animal.n.01", label="Root Synset", info="The WordNet synset ID to start from (e.g., animal.n.01).")
                depth_input = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Max Depth", info="How deep to traverse the tree structure.")

            filter_input = gr.Checkbox(label="Filter using ImageNet 1k list", info="If checked, only synsets present in the ImageNet 1k dataset will be included.")
            output_path_tree = gr.Textbox(value="wildcards_imagenet.yaml", label="Output Filename")

            with gr.Row():
                btn_preview_tree = gr.Button("Preview YAML", variant="secondary")
                btn_save_tree = gr.Button("Generate & Save File", variant="primary")

            out_preview_tree = gr.Code(language="yaml", label="Preview")
            out_status_tree = gr.Textbox(label="Status", interactive=False)

            btn_preview_tree.click(preview_tree, inputs=[root_input, depth_input, filter_input], outputs=[out_preview_tree])
            btn_save_tree.click(save_tree, inputs=[root_input, depth_input, filter_input, output_path_tree], outputs=[out_status_tree])

        # --- COCO Tab ---
        with gr.TabItem("COCO"):
            gr.Markdown("### COCO Dataset Hierarchy")
            gr.Markdown("Generate hierarchy from COCO categories. It groups categories under their supercategories.")

            gr.Markdown("**Note:** This may download the COCO annotations file if not present locally.")

            output_path_coco = gr.Textbox(value="wildcards_coco.yaml", label="Output Filename")

            with gr.Row():
                btn_preview_coco = gr.Button("Preview YAML", variant="secondary")
                btn_save_coco = gr.Button("Generate & Save File", variant="primary")

            out_preview_coco = gr.Code(language="yaml", label="Preview")
            out_status_coco = gr.Textbox(label="Status", interactive=False)

            btn_preview_coco.click(preview_coco, outputs=[out_preview_coco])
            btn_save_coco.click(save_coco, inputs=[output_path_coco], outputs=[out_status_coco])

        # --- Open Images Tab ---
        with gr.TabItem("Open Images"):
            gr.Markdown("### Open Images Dataset Hierarchy")
            gr.Markdown("Generate hierarchy from Open Images class descriptions and hierarchy JSON.")

            gr.Markdown("**Note:** This will download the class descriptions and hierarchy JSON if not present.")

            output_path_oi = gr.Textbox(value="wildcards_openimages.yaml", label="Output Filename")

            with gr.Row():
                btn_preview_oi = gr.Button("Preview YAML", variant="secondary")
                btn_save_oi = gr.Button("Generate & Save File", variant="primary")

            out_preview_oi = gr.Code(language="yaml", label="Preview")
            out_status_oi = gr.Textbox(label="Status", interactive=False)

            btn_preview_oi.click(preview_oi, outputs=[out_preview_oi])
            btn_save_oi.click(save_oi, inputs=[output_path_oi], outputs=[out_status_oi])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
