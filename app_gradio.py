#!/usr/bin/env python3
"""
Gradio GUI for Hierarchy Generator
"""

import gradio as gr
import app
import yaml
import os
import logging

# Set up logging
logger = logging.getLogger("gui")

def format_yaml_preview(data, max_lines=1000):
    yaml_str = yaml.dump(data, sort_keys=False, default_flow_style=False, indent=4)
    lines = yaml_str.split('\n')
    if len(lines) > max_lines:
        return '\n'.join(lines[:max_lines]) + f"\n\n... (Truncated. Total lines: {len(lines)})"
    return yaml_str

def safe_execution(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        return f"Error: {str(e)}"

def load_file_content(file_path):
    if not file_path:
        return ""
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"

# --- logic wrapper ---

def get_imagenet_filter_set(filter_mode):
    if filter_mode == "ImageNet 1k":
        return app.load_imagenet_1k_set()
    elif filter_mode == "ImageNet 21k":
        return app.load_imagenet_21k_set()
    return None

def generate_hierarchy(mode, wnid_text, root_synset, filter_mode, max_depth):
    # Ensure depth is int
    max_depth = int(max_depth)

    if mode == "ImageNet (Custom List)":
        if not wnid_text.strip():
            raise ValueError("Please enter at least one WNID.")
        wnids = [line.strip() for line in wnid_text.split('\n') if line.strip()]
        h = app.generate_imagenet_wnid_hierarchy(wnids, max_depth)

    elif mode == "ImageNet (Tree)":
        filter_set = get_imagenet_filter_set(filter_mode)
        h = app.generate_imagenet_tree_hierarchy(root_synset, max_depth, filter_set)

    elif mode == "COCO":
        h = app.generate_coco_hierarchy(max_depth)

    elif mode == "Open Images":
        h = app.generate_openimages_hierarchy(max_depth)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return app.convert_to_wildcard_format(h)

def on_preview(mode, wnid_text, root_synset, filter_mode, max_depth):
    try:
        # Fix 1: Handle None inputs
        wnid_text = wnid_text or ""
        root_synset = root_synset or "entity.n.01"

        # Fix 3: Validation
        if mode == "ImageNet (Custom List)" and not wnid_text.strip():
            return "# Error: Please enter WNIDs", "❌ Error: No WNIDs provided"

        h = generate_hierarchy(mode, wnid_text, root_synset, filter_mode, max_depth)

        # Stats
        if isinstance(h, list):
            count = len(h)
            msg = f"Generated List with {count} items."
        else:
            # Count leaves roughly
            leaves = app.extract_all_leaves(h)
            msg = f"Generated Hierarchy with ~{len(leaves)} leaf items."

        return format_yaml_preview(h), msg
    except Exception as e:
        return f"Error: {e}", f"Error: {e}"

def on_save(mode, wnid_text, root_synset, filter_mode, max_depth, output_path):
    try:
        # Handle None inputs
        wnid_text = wnid_text or ""
        root_synset = root_synset or "entity.n.01"

        # Validation
        if mode == "ImageNet (Custom List)" and not wnid_text.strip():
            return "❌ Error: No WNIDs provided"

        h = generate_hierarchy(mode, wnid_text, root_synset, filter_mode, max_depth)
        app.save_hierarchy(h, output_path)
        return f"Successfully saved to {output_path}"
    except Exception as e:
        return f"Error: {e}"

# --- GUI Construction ---

with gr.Blocks(title="Hierarchy Generator") as demo:
    gr.Markdown("# Wildcard Hierarchy Generator")
    gr.Markdown("Generate YAML hierarchies for ImageNet, COCO, and Open Images datasets.")

    # Main Selector
    mode = gr.Radio(
        ["ImageNet (Custom List)", "ImageNet (Tree)", "COCO", "Open Images"],
        label="Dataset Mode",
        value="ImageNet (Tree)"
    )

    # Common Controls
    with gr.Row():
        max_depth = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Max Depth",
                              info="Flatten hierarchy into list at this depth.")
        output_path = gr.Textbox(value="wildcards_output.yaml", label="Output Filename")

    # --- Mode Specific Groups ---

    # ImageNet Custom List
    with gr.Group(visible=False) as grp_custom:
        gr.Markdown("### ImageNet Custom List")
        gr.Markdown("Provide a list of WNIDs to build a hierarchy bottom-up.")
        with gr.Row():
            with gr.Column(scale=2):
                wnid_input = gr.Textbox(lines=10, label="WNIDs (one per line)", placeholder="n02084071...")
            with gr.Column(scale=1):
                file_upload = gr.File(label="Upload List", type="filepath")

    # ImageNet Tree
    with gr.Group(visible=True) as grp_tree:
        gr.Markdown("### ImageNet Tree")
        gr.Markdown("Recursive top-down generation.")

        filter_mode = gr.Radio(["All WordNet", "ImageNet 1k", "ImageNet 21k"],
                               label="Filter Preset", value="All WordNet")

        with gr.Accordion("Advanced Settings", open=False):
            root_input = gr.Textbox(value="entity.n.01", label="Root Synset",
                                    info="Leave as entity.n.01 for full WordNet (filtered).")

    # COCO
    with gr.Group(visible=False) as grp_coco:
        gr.Markdown("### COCO")
        gr.Markdown("Generates COCO hierarchy. Depth 1 = Flat List.")

    # Open Images
    with gr.Group(visible=False) as grp_oi:
        gr.Markdown("### Open Images")
        gr.Markdown("Generates Open Images hierarchy.")

    # Actions
    with gr.Row():
        btn_preview = gr.Button("Preview YAML", variant="secondary")
        btn_save = gr.Button("Generate & Save", variant="primary")

    # Output (Reorganized)
    # Fix 2 & 4: Status above preview, improved feedback
    out_status = gr.Textbox(label="Status", interactive=False, show_label=True)

    with gr.Accordion("Preview Output", open=True):
        out_preview = gr.Code(language="yaml", label="Generated YAML")

    # Fix 4: Add Examples
    gr.Examples(
        examples=[
            ["ImageNet (Tree)", "", "entity.n.01", "ImageNet 1k", 2],
            ["COCO", "", "", "All WordNet", 1],
        ],
        inputs=[mode, wnid_input, root_input, filter_mode, max_depth],
    )

    # --- Interaction Logic ---

    def update_visibility(selected_mode):
        return {
            grp_custom: gr.update(visible=(selected_mode == "ImageNet (Custom List)")),
            grp_tree: gr.update(visible=(selected_mode == "ImageNet (Tree)")),
            grp_coco: gr.update(visible=(selected_mode == "COCO")),
            grp_oi: gr.update(visible=(selected_mode == "Open Images"))
        }

    mode.change(update_visibility, inputs=[mode], outputs=[grp_custom, grp_tree, grp_coco, grp_oi])

    file_upload.change(load_file_content, inputs=[file_upload], outputs=[wnid_input])

    btn_preview.click(on_preview,
                      inputs=[mode, wnid_input, root_input, filter_mode, max_depth],
                      outputs=[out_preview, out_status])

    btn_save.click(on_save,
                   inputs=[mode, wnid_input, root_input, filter_mode, max_depth, output_path],
                   outputs=[out_status])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
