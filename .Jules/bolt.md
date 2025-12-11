## 2024-05-23 - Caching Static Dataset Loads
**Learning:** Large static datasets (ImageNet, COCO) were being re-loaded from disk on every operation, causing significant latency in the GUI. Caching these in memory is a huge win.
**Action:** Always check if static resources are being re-loaded in hot paths and use `@lru_cache`.
