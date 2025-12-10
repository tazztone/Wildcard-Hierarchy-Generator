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

def validate_inputs(mode: str, strategy: str, wnid_text: str, root_synset: str,
                   max_depth: int) -> Tuple[bool, str]:
    """Validate inputs based on selected mode and strategy."""
    # Note: Inputs should be normalized before calling this, but we double-check here
    wnid_text = (wnid_text or "").strip()
    root_synset = (root_synset or "entity.n.01").strip()

    # Mode-specific validation
    if mode == "ImageNet":
        if strategy == "Custom List":
            if not wnid_text:
                return False, "⚠️ Please enter at least one WNID in the custom list"
            wnids = [line.strip() for line in wnid_text.split('\n') if line.strip()]
            if len(wnids) == 0:
                return False, "⚠️ No valid WNIDs found"
            return True, f"✓ {len(wnids)} WNIDs ready to process"
        else: # Recursive (Standard)
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

def generate_hierarchy(mode: str, strategy: str, wnid_text: str, root_synset: str,
                      filter_mode: str, max_depth: int,
                      strict_filter: bool = True, blacklist: bool = False,
                      hypernym_depth: int = 0) -> Any:
    """Generate hierarchy based on mode and parameters."""
    max_depth = int(max_depth)
    wnid_text = (wnid_text or "").strip()
    root_synset = (root_synset or "entity.n.01").strip()

    if mode == "ImageNet":
        if strategy == "Custom List":
            wnids = [line.strip() for line in wnid_text.split('\n') if line.strip()]
            if not wnids:
                raise ValueError("No valid WNIDs provided")
            hyp_depth = int(hypernym_depth) if hypernym_depth > 0 else None
            return app.generate_imagenet_wnid_hierarchy(wnids, max_depth, hyp_depth)
        else:
            filter_set = get_imagenet_filter_set(filter_mode)
            return app.generate_imagenet_tree_hierarchy(root_synset, max_depth, filter_set, strict_filter, blacklist)

    elif mode == "COCO":
        return app.generate_coco_hierarchy(max_depth)

    elif mode == "Open Images":
        return app.generate_openimages_hierarchy(max_depth)

    raise ValueError(f"Unknown mode: {mode}")

# ============================================================================
# UI ACTION HANDLERS
# ============================================================================

def on_preview(mode: str, strategy: str, wnid_text: str, root_synset: str,
               filter_mode: str, max_depth: int,
               strict_filter: bool, blacklist: bool, hypernym_depth: int) -> Tuple[str, str]:
    """Handle preview button click."""
    try:
        # Normalize inputs
        wnid_text = (wnid_text or "").strip()
        root_synset = (root_synset or "entity.n.01").strip()

        # Validate inputs
        is_valid, message = validate_inputs(mode, strategy, wnid_text, root_synset, max_depth)
        if not is_valid:
            return "# " + message, message

        # Generate hierarchy
        logger.info(f"Generating preview for mode: {mode}, strategy: {strategy}")
        hierarchy = generate_hierarchy(mode, strategy, wnid_text, root_synset, filter_mode, max_depth,
                                     strict_filter, blacklist, hypernym_depth)
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

def on_save(mode: str, strategy: str, wnid_text: str, root_synset: str,
            filter_mode: str, max_depth: int, output_path: str,
            strict_filter: bool, blacklist: bool, hypernym_depth: int) -> str:
    """Handle save button click."""
    try:
        # Normalize inputs
        wnid_text = (wnid_text or "").strip()
        root_synset = (root_synset or "entity.n.01").strip()

        # Validate inputs
        is_valid, message = validate_inputs(mode, strategy, wnid_text, root_synset, max_depth)
        if not is_valid:
            return message

        # Validate output path
        output_path = (output_path or "wildcards_output.yaml").strip()
        if not output_path.endswith('.yaml') and not output_path.endswith('.yml'):
            output_path += '.yaml'

        # Generate and save
        logger.info(f"Generating and saving to: {output_path}")
        hierarchy = generate_hierarchy(mode, strategy, wnid_text, root_synset, filter_mode, max_depth,
                                     strict_filter, blacklist, hypernym_depth)
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

def get_mode_from_index(index: int) -> str:
    """Get mode name from tab index."""
    modes = ["ImageNet", "COCO", "Open Images"]
    return modes[index] if index < len(modes) else "Unknown"

def on_tab_change(evt: gr.SelectData) -> Tuple[str, str]:
    """Handle tab change to update mode and filename."""
    mode_name = get_mode_from_index(evt.index)
    logger.info(f"Tab changed to index {evt.index}, mode: {mode_name}")

    # Auto-update filename
    safe_name = mode_name.lower().replace(" ", "")
    new_filename = f"wildcards_{safe_name}.yaml"
    return mode_name, new_filename

# ============================================================================
# UI CONSTRUCTION
# ============================================================================

def create_ui() -> gr.Blocks:
    """Create and configure the Gradio interface."""

    common_roots = [
        ("Everything (entity.n.01)", "entity.n.01"),
        ("Physical Objects (physical_entity.n.01)", "physical_entity.n.01"),
        ("Organisms (organism.n.01)", "organism.n.01"),
        ("Animals (animal.n.01)", "animal.n.01"),
        ("People (person.n.01)", "person.n.01"),
        ("Artifacts/Objects (artifact.n.01)", "artifact.n.01"),
        ("Abstract Concepts (abstraction.n.06)", "abstraction.n.06"),
    ]

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
                info="How deep to traverse the tree. Reaching the limit flattens all children below it into a list."
            )
            output_path = gr.Textbox(
                value="wildcards_output.yaml",
                label="💾 Output Filename",
                placeholder="output.yaml"
            )

        # Mode Selection with Tabs
        with gr.Tabs() as tabs:

            # Tab 1: ImageNet (Combined)
            with gr.Tab("🖼️ ImageNet", id=0):
                gr.Markdown("""
                Generate a hierarchy from WordNet (ImageNet's source).
                Choose between starting from a root category or providing a custom list of IDs.
                """)

                # Strategy Selection
                im_strategy = gr.Radio(
                    choices=["Recursive (from Root)", "Custom List"],
                    value="Recursive (from Root)",
                    label="Generation Method",
                    info="Choose 'Recursive' to build a tree from a category, or 'Custom List' to use specific IDs."
                )

                # SECTION: Recursive (Tree)
                with gr.Group(visible=True) as group_recursive:
                    with gr.Row():
                        im_root = gr.Dropdown(
                            choices=common_roots,
                            value="entity.n.01",
                            allow_custom_value=True,
                            label="🌱 Root Synset",
                            info="The starting category. Type a custom synset (e.g. 'dog.n.01') if needed."
                        )
                        im_filter = gr.Radio(
                            choices=["All WordNet", "ImageNet 1k", "ImageNet 21k"],
                            value="All WordNet",
                            label="🔍 Filter Preset",
                            info="Restrict results to a specific dataset version."
                        )
                    with gr.Row():
                        im_strict = gr.Checkbox(
                            value=True,
                            label="Strict Synset Filtering",
                            info="Exclude secondary meanings (e.g. slang terms) that pollute the hierarchy."
                        )
                        im_blacklist = gr.Checkbox(
                            value=False,
                            label="Exclude Abstract Categories",
                            info="Exclude high-level abstract categories like 'communication', 'measure', etc."
                        )

                    with gr.Accordion("ℹ️ Help: Understanding Filters", open=False):
                        gr.Markdown("""
                        - **All WordNet**: Uses the full WordNet English lexical database (largest).
                        - **ImageNet 1k**: Restricts the tree to only include synsets found in the ImageNet-1K dataset.
                        - **ImageNet 21k**: Restricts the tree to only include synsets found in the ImageNet-21K dataset.
                        - **Strict Synset Filtering**: Only includes nodes where the synset is the primary meaning of the word. Fixes issues like 'old man' (slang) appearing under 'communication'.
                        """)

                # SECTION: Custom List
                with gr.Accordion("📋 Custom List Settings", open=False, visible=False) as group_custom:
                    gr.Markdown("Build a hierarchy from specific WordNet IDs (WNIDs).")
                    with gr.Row():
                        with gr.Column(scale=3):
                            im_wnid_input = gr.Textbox(
                                lines=12,
                                label="WNIDs (one per line)",
                                placeholder="n02084071\nn02121808\nn01503061\n...",
                                info="Enter WordNet IDs to include."
                            )
                        with gr.Column(scale=1):
                            file_upload = gr.File(
                                label="📁 Upload WNID List",
                                file_types=[".txt", ".csv"],
                                type="filepath"
                            )
                            im_hypernym_depth = gr.Slider(
                                minimum=0, maximum=10, step=1, value=0,
                                label="Max Hypernym Depth",
                                info="Limit the height of the tree above the leaves (0 = Full Path). Useful to create a 'forest' of categories."
                            )
                            btn_clear_wnid = gr.Button("🗑️ Clear List", variant="secondary", size="sm")

                            gr.Markdown(
                                "Need WNIDs? [Search ImageNet](https://image-net.org/challenges/LSVRC/2012/browse-synsets) or [WordNet](http://wordnetweb.princeton.edu/perl/webwn).",
                                line_breaks=True
                            )

                # Visibility Logic for Strategy
                def update_visibility(strat):
                    is_custom = (strat == "Custom List")
                    return {
                        group_recursive: gr.update(visible=not is_custom),
                        group_custom: gr.update(visible=is_custom, open=True) # Auto-open when selected
                    }

                im_strategy.change(
                    update_visibility,
                    inputs=[im_strategy],
                    outputs=[group_recursive, group_custom]
                )

                # State trackers
                mode_imagenet = gr.State("ImageNet")

            # Tab 2: COCO
            with gr.Tab("📷 COCO Dataset", id=1):
                gr.Markdown("""
                Generate hierarchy for COCO (Common Objects in Context) dataset.
                Depth 1 produces a flat list of 80 categories.
                """)
                mode_coco = gr.State("COCO")
                # Dummy states for COCO dispatch alignment
                wnid_coco = gr.State("")
                root_coco = gr.State("entity.n.01")
                filter_coco = gr.State("All WordNet")
                strat_coco = gr.State("Default")

            # Tab 3: Open Images
            with gr.Tab("🖼️ Open Images", id=2):
                gr.Markdown("""
                Generate hierarchy for Open Images dataset (600+ categories).
                """)
                mode_oi = gr.State("Open Images")
                # Dummy states
                wnid_oi = gr.State("")
                root_oi = gr.State("entity.n.01")
                filter_oi = gr.State("All WordNet")
                strat_oi = gr.State("Default")

        # Current mode tracker
        current_mode = gr.State("ImageNet")

        # Validation Status
        validation_status = gr.Textbox(
            label="🔔 Validation Status",
            value="✓ Ready to generate",
            interactive=False,
            show_label=True,
            show_copy_button=True
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
            lines=2,
            show_copy_button=True
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
                    ["Recursive (from Root)", "entity.n.01", "ImageNet 1k", 3],
                    ["Recursive (from Root)", "animal.n.01", "All WordNet", 2],
                ],
                inputs=[im_strategy, im_root, im_filter, max_depth],
                label="Try these ImageNet configurations"
            )

        # ====================================================================
        # EVENT HANDLERS
        # ====================================================================

        tabs.select(on_tab_change, outputs=[current_mode, output_path])

        # File upload handler
        file_upload.change(
            load_file_content,
            inputs=[file_upload],
            outputs=[im_wnid_input]
        )

        # Clear button handler
        btn_clear_wnid.click(
            lambda: "",
            outputs=[im_wnid_input]
        )

        # Real-time validation (ImageNet)
        def run_validation_im(strat, wnid, root):
            valid, msg = validate_inputs("ImageNet", strat, wnid, root, 1)
            return msg

        im_input_triggers = [im_strategy, im_wnid_input, im_root]
        for trig in im_input_triggers:
            trig.change(run_validation_im, inputs=[im_strategy, im_wnid_input, im_root], outputs=[validation_status])

        # Dispatch logic
        def dispatch_preview(mode,
                           im_strat, im_rt, im_filt, im_wnid, im_strict, im_blacklist, im_hyp_depth,
                           co_strat, co_rt, co_filt, co_wnid,
                           oi_strat, oi_rt, oi_filt, oi_wnid,
                           depth):

            # Map inputs based on mode
            if mode == "ImageNet":
                return on_preview(mode, im_strat, im_wnid, im_rt, im_filt, depth, im_strict, im_blacklist, im_hyp_depth)
            elif mode == "COCO":
                # Pass dummies for strict/blacklist/hyp_depth
                return on_preview(mode, co_strat, co_wnid, co_rt, co_filt, depth, True, False, 0)
            elif mode == "Open Images":
                return on_preview(mode, oi_strat, oi_wnid, oi_rt, oi_filt, depth, True, False, 0)
            return "# Error", "Unknown mode"

        def dispatch_save(mode,
                          im_strat, im_rt, im_filt, im_wnid, im_strict, im_blacklist, im_hyp_depth,
                          co_strat, co_rt, co_filt, co_wnid,
                          oi_strat, oi_rt, oi_filt, oi_wnid,
                          depth, path):
             if mode == "ImageNet":
                 return on_save(mode, im_strat, im_wnid, im_rt, im_filt, depth, path, im_strict, im_blacklist, im_hyp_depth)
             elif mode == "COCO":
                 return on_save(mode, co_strat, co_wnid, co_rt, co_filt, depth, path, True, False, 0)
             elif mode == "Open Images":
                 return on_save(mode, oi_strat, oi_wnid, oi_rt, oi_filt, depth, path, True, False, 0)
             return "Error: Unknown mode"

        all_inputs = [
            current_mode,
            # ImageNet
            im_strategy, im_root, im_filter, im_wnid_input, im_strict, im_blacklist, im_hypernym_depth,
            # COCO (dummies)
            strat_coco, root_coco, filter_coco, wnid_coco,
            # OI (dummies)
            strat_oi, root_oi, filter_oi, wnid_oi,
            # Global
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
