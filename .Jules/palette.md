## 2025-01-26 - Gradio Context Switching
**Learning:** Using `tabs.select` with `gr.SelectData` allows for powerful context-aware state updates (like auto-renaming files) without complex listeners.
**Action:** Look for other tab-based interfaces to apply "smart defaults" when context changes.

## 2025-01-27 - Gradio Progress Injection
**Learning:** Adding `progress=gr.Progress()` to event handlers (even nested dispatch functions) provides immediate visual feedback without complex UI state management.
**Action:** Use `gr.Progress` for all long-running Gradio operations instead of manual status text updates.
