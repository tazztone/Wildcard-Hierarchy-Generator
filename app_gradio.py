#!/usr/bin/env python3
"""
Gradio GUI for Hierarchy Generator - Rewritten for better UX and maintainability
"""

import gradio as gr
import app
import yaml
import os
import logging
from typing import Dict, Tuple, Optional, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gui")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_yaml_preview(data: Any, max_lines: int = 1000) -> str:
    """Format data as YAML with optional truncation."""
    try:
        yaml_str = yaml.dump(data, sort_keys=False, default_flow_style=False, indent=2)
        lines = yaml_str.split('\n')
        if len(lines) > max_lines:
            truncated = '\n'.join(lines[:max_lines])
            return f"{truncated}\n\n... (Truncated. Showing {max_lines}/{len(lines)} lines)"
        return yaml_str
    except Exception as e:
        logger.error(f"YAML formatting error: {e}")
        return f"# Error formatting YAML: {str(e)}"

def load_file_content(file_path: Optional[str]) -> str:
    """Load content from uploaded file."""
    if not file_path or not os.path.exists(file_path):
        return ""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        logger.info(f"Loaded file: {file_path}")
        return content
    except Exception as e:
        logger.error(f"File read error: {e}")
        return f"# Error reading file: {str(e)}"

def count_hierarchy_items(hierarchy: Any) -> int:
    """Count items in hierarchy structure."""
    if isinstance(hierarchy, list):
        return len(hierarchy)
    elif isinstance(hierarchy, dict):
        return len(app.extract_all_leaves(hierarchy))
    return 0

# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_inputs(mode: str, wnid_text: str, root_synset: str,
                   max_depth: int) -> Tuple[bool, str]:
    """Validate inputs based on selected mode."""
    # Normalize inputs
    wnid_text = (wnid_text or "").strip()
    root_synset = (root_synset or "entity.n.01").strip()

    # Mode-specific validation
    if mode == "ImageNet (Custom List)":
        if not wnid_text:
            return False, "⚠️ Please enter at least one WNID"
        wnids = [line.strip() for line in wnid_text.split('\n') if line.strip()]
        if len(wnids) == 0:
            return False, "⚠️ No valid WNIDs found"
        return True, f"✓ {len(wnids)} WNIDs ready to process"

    elif mode == "ImageNet (Tree)":
        if not root_synset:
            return False, "⚠️ Root synset cannot be empty"
        return True, f"✓ Ready to generate tree from '{root_synset}'"

    elif mode in ["COCO", "Open Images"]:
        return True, f"✓ Ready to generate {mode} hierarchy"

    return False, "⚠️ Unknown mode selected"

# ============================================================================
# CORE GENERATION FUNCTIONS
# ============================================================================

def get_imagenet_filter_set(filter_mode: str) -> Optional[set]:
    """Get appropriate ImageNet filter set."""
    if filter_mode == "ImageNet 1k":
        return app.load_imagenet_1k_set()
    elif filter_mode == "ImageNet 21k":
        return app.load_imagenet_21k_set()
    return None

def generate_hierarchy(mode: str, wnid_text: str, root_synset: str,
                      filter_mode: str, max_depth: int) -> Any:
    """Generate hierarchy based on mode and parameters."""
    max_depth = int(max_depth)
    wnid_text = (wnid_text or "").strip()
    root_synset = (root_synset or "entity.n.01").strip()

    if mode == "ImageNet (Custom List)":
        wnids = [line.strip() for line in wnid_text.split('\n') if line.strip()]
        if not wnids:
            raise ValueError("No valid WNIDs provided")
        return app.generate_imagenet_wnid_hierarchy(wnids, max_depth)

    elif mode == "ImageNet (Tree)":
        filter_set = get_imagenet_filter_set(filter_mode)
        return app.generate_imagenet_tree_hierarchy(root_synset, max_depth, filter_set)

    elif mode == "COCO":
        return app.generate_coco_hierarchy(max_depth)

    elif mode == "Open Images":
        return app.generate_openimages_hierarchy(max_depth)

    raise ValueError(f"Unknown mode: {mode}")

# ============================================================================
# UI ACTION HANDLERS
# ============================================================================

def on_preview(mode: str, wnid_text: str, root_synset: str,
               filter_mode: str, max_depth: int) -> Tuple[str, str]:
    """Handle preview button click."""
    try:
        # Validate inputs
        is_valid, message = validate_inputs(mode, wnid_text, root_synset, max_depth)
        if not is_valid:
            return "# " + message, message

        # Generate hierarchy
        logger.info(f"Generating preview for mode: {mode}")
        hierarchy = generate_hierarchy(mode, wnid_text, root_synset, filter_mode, max_depth)
        wildcard_format = app.convert_to_wildcard_format(hierarchy)

        # Format output
        yaml_output = format_yaml_preview(wildcard_format)
        item_count = count_hierarchy_items(wildcard_format)

        status = f"✅ Preview generated successfully\n"
        status += f"📊 Contains {item_count} items"
        if max_depth > 1:
            status += f" (depth: {max_depth})"

        logger.info(f"Preview completed: {item_count} items")
        return yaml_output, status

    except Exception as e:
        error_msg = f"❌ Generation failed: {str(e)}"
        logger.error(error_msg)
        return f"# Error\n# {str(e)}", error_msg

def on_save(mode: str, wnid_text: str, root_synset: str,
            filter_mode: str, max_depth: int, output_path: str) -> str:
    """Handle save button click."""
    try:
        # Validate inputs
        is_valid, message = validate_inputs(mode, wnid_text, root_synset, max_depth)
        if not is_valid:
            return message

        # Validate output path
        output_path = (output_path or "wildcards_output.yaml").strip()
        if not output_path.endswith('.yaml') and not output_path.endswith('.yml'):
            output_path += '.yaml'

        # Generate and save
        logger.info(f"Generating and saving to: {output_path}")
        hierarchy = generate_hierarchy(mode, wnid_text, root_synset, filter_mode, max_depth)
        wildcard_format = app.convert_to_wildcard_format(hierarchy)
        app.save_hierarchy(wildcard_format, output_path)

        item_count = count_hierarchy_items(wildcard_format)
        status = f"✅ Successfully saved to '{output_path}'\n"
        status += f"📊 {item_count} items written"

        logger.info(f"Save completed: {output_path}")
        return status

    except Exception as e:
        error_msg = f"❌ Save failed: {str(e)}"
        logger.error(error_msg)
        return error_msg

def on_validate_input(mode: str, wnid_text: str, root_synset: str,
                     max_depth: int) -> str:
    """Real-time input validation feedback."""
    is_valid, message = validate_inputs(mode, wnid_text, root_synset, max_depth)
    return message

# ============================================================================
# UI CONSTRUCTION
# ============================================================================

def create_ui() -> gr.Blocks:
    """Create and configure the Gradio interface."""

    with gr.Blocks(
        title="Wildcard Hierarchy Generator",
        theme=gr.themes.Soft(),
        css=".gradio-container {max-width: 1200px !important}"
    ) as demo:

        # Header
        gr.Markdown("""
        # 🌳 Wildcard Hierarchy Generator
        Generate structured YAML hierarchies from ImageNet, COCO, and Open Images datasets.
        """)

        # Global Controls
        with gr.Row():
            max_depth = gr.Slider(
                minimum=1, maximum=10, value=3, step=1,
                label="🎯 Max Depth",
                info="Hierarchy depth (1 = flat list)"
            )
            output_path = gr.Textbox(
                value="wildcards_output.yaml",
                label="💾 Output Filename",
                placeholder="output.yaml"
            )

        # Mode Selection with Tabs
        with gr.Tabs() as tabs:

            # Tab 1: ImageNet Custom List
            with gr.Tab("📋 ImageNet Custom List", id=0):
                gr.Markdown("""
                Build a hierarchy from specific WordNet IDs (WNIDs).
                Enter WNIDs one per line, or upload a text file.
                """)

                with gr.Row():
                    with gr.Column(scale=3):
                        wnid_input = gr.Textbox(
                            lines=12,
                            label="WNIDs (one per line)",
                            placeholder="n02084071\\nn02121808\\nn01503061\\n...",
                            info="Enter WordNet IDs to include in hierarchy"
                        )
                    with gr.Column(scale=1):
                        file_upload = gr.File(
                            label="📁 Upload WNID List",
                            file_types=[".txt", ".csv"],
                            type="filepath"
                        )
                        gr.Markdown("""
                        **Quick Start:**
                        1. Paste WNIDs
                        2. Set depth
                        3. Click Preview
                        """)

                mode_custom = gr.State("ImageNet (Custom List)")
                filter_custom = gr.State("All WordNet")
                root_custom = gr.State("entity.n.01")

            # Tab 2: ImageNet Tree
            with gr.Tab("🌲 ImageNet Tree", id=1):
                gr.Markdown("""
                Generate a complete hierarchy starting from a root synset.
                Use filters to limit to specific ImageNet versions.
                """)

                root_input = gr.Textbox(
                    value="entity.n.01",
                    label="🌱 Root Synset",
                    info="Starting point for tree generation (default: entity.n.01)",
                    placeholder="entity.n.01"
                )

                filter_mode = gr.Radio(
                    choices=["All WordNet", "ImageNet 1k", "ImageNet 21k"],
                    value="All WordNet",
                    label="🔍 Filter Preset",
                    info="Restrict to specific ImageNet version"
                )

                with gr.Accordion("ℹ️ About Filters", open=False):
                    gr.Markdown("""
                    - **All WordNet**: Full WordNet hierarchy (largest)
                    - **ImageNet 1k**: Only synsets in ImageNet-1K dataset
                    - **ImageNet 21k**: Only synsets in ImageNet-21K dataset
                    """)

                mode_tree = gr.State("ImageNet (Tree)")
                wnid_tree = gr.State("")

            # Tab 3: COCO
            with gr.Tab("📷 COCO Dataset", id=2):
                gr.Markdown("""
                Generate hierarchy for COCO (Common Objects in Context) dataset.
                Depth 1 produces a flat list of 80 categories.
                """)

                gr.Markdown("""
                **COCO Info:**
                - 80 object categories
                - Commonly used for object detection
                - Depth 1 recommended (categories are already flat)
                """)

                mode_coco = gr.State("COCO")
                wnid_coco = gr.State("")
                root_coco = gr.State("entity.n.01")
                filter_coco = gr.State("All WordNet")

            # Tab 4: Open Images
            with gr.Tab("🖼️ Open Images", id=3):
                gr.Markdown("""
                Generate hierarchy for Open Images dataset.
                Contains 600+ object categories.
                """)

                gr.Markdown("""
                **Open Images Info:**
                - 600+ diverse categories
                - Hierarchical structure
                - Good for broad classification tasks
                """)

                mode_oi = gr.State("Open Images")
                wnid_oi = gr.State("")
                root_oi = gr.State("entity.n.01")
                filter_oi = gr.State("All WordNet")

        # Current mode tracker
        current_mode = gr.State("ImageNet (Custom List)")

        # Validation Status
        validation_status = gr.Textbox(
            label="🔔 Validation Status",
            value="✓ Ready to generate",
            interactive=False,
            show_label=True
        )

        # Action Buttons
        with gr.Row():
            btn_preview = gr.Button(
                "👁️ Preview YAML",
                variant="secondary",
                size="lg"
            )
            btn_save = gr.Button(
                "💾 Generate & Save",
                variant="primary",
                size="lg"
            )

        # Output Section
        gr.Markdown("---")
        gr.Markdown("## 📤 Output")

        out_status = gr.Textbox(
            label="Status",
            value="",
            interactive=False,
            show_label=True,
            lines=2
        )

        with gr.Accordion("📄 YAML Preview", open=True):
            out_preview = gr.Code(
                language="yaml",
                label="Generated YAML",
                lines=20
            )

        # Examples
        with gr.Accordion("💡 Examples", open=False):
            gr.Examples(
                examples=[
                    [2, "ImageNet 1k"],  # Tree mode example
                    [1, "All WordNet"],  # Flat list
                    [3, "ImageNet 21k"], # Deep hierarchy
                ],
                inputs=[max_depth, filter_mode],
                label="Try these configurations for ImageNet Tree"
            )

        # ====================================================================
        # EVENT HANDLERS
        # ====================================================================

        def get_current_inputs(tab_index):
            """Return appropriate inputs based on active tab."""
            modes = [
                ("ImageNet (Custom List)", wnid_input, root_custom, filter_custom),
                ("ImageNet (Tree)", wnid_tree, root_input, filter_mode),
                ("COCO", wnid_coco, root_coco, filter_coco),
                ("Open Images", wnid_oi, root_oi, filter_oi)
            ]
            return modes[tab_index]

        # Tab change handler
        def on_tab_change(evt: gr.SelectData):
            mode_info = get_current_inputs(evt.index)
            logger.info(f"Tab changed to index {evt.index}, mode: {mode_info[0]}")
            return mode_info[0]  # Return mode name

        tabs.select(on_tab_change, outputs=[current_mode])

        # File upload handler
        file_upload.change(
            load_file_content,
            inputs=[file_upload],
            outputs=[wnid_input]
        )

        # Real-time validation (for custom list tab)
        wnid_input.change(
            lambda text: validate_inputs("ImageNet (Custom List)", text, "", 1)[1],
            inputs=[wnid_input],
            outputs=[validation_status]
        )

        # Dispatch logic to handle multiple tabs with one button
        def dispatch_preview(mode,
                           wc, rc, fc,
                           wt, rt, ft,
                           wco, rco, fco,
                           woi, roi, foi,
                           depth):
            if mode == "ImageNet (Custom List)":
                return on_preview(mode, wc, rc, fc, depth)
            elif mode == "ImageNet (Tree)":
                return on_preview(mode, wt, rt, ft, depth)
            elif mode == "COCO":
                return on_preview(mode, wco, rco, fco, depth)
            elif mode == "Open Images":
                return on_preview(mode, woi, roi, foi, depth)
            return "# Error", "Unknown mode"

        def dispatch_save(mode,
                          wc, rc, fc,
                          wt, rt, ft,
                          wco, rco, fco,
                          woi, roi, foi,
                          depth, path):
             if mode == "ImageNet (Custom List)":
                 return on_save(mode, wc, rc, fc, depth, path)
             elif mode == "ImageNet (Tree)":
                 return on_save(mode, wt, rt, ft, depth, path)
             elif mode == "COCO":
                 return on_save(mode, wco, rco, fco, depth, path)
             elif mode == "Open Images":
                 return on_save(mode, woi, roi, foi, depth, path)
             return "Error: Unknown mode"

        all_inputs = [
            current_mode,
            wnid_input, root_custom, filter_custom,
            wnid_tree, root_input, filter_mode,
            wnid_coco, root_coco, filter_coco,
            wnid_oi, root_oi, filter_oi,
            max_depth
        ]

        btn_preview.click(
            dispatch_preview,
            inputs=all_inputs,
            outputs=[out_preview, out_status]
        )

        btn_save.click(
            dispatch_save,
            inputs=all_inputs + [output_path],
            outputs=[out_status]
        )

    return demo

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
