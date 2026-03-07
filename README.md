# KNNIFE

*Riccardo De Vidi, Francesco Borsatti, Valentina Zaccaria, Davide Sartor, Gian Antonio Susto*

---

Code for the paper *KNNIFE: k-Nearest Neighbors Isolation Forest Interpretability*.

KNNIFE and sKNNIFE are feature importance methods for Isolation Forest-based anomaly detectors (IF, EIF, DIF). They assign importance scores to features by using k-nearest neighbor distances at each tree node split, and are benchmarked against DIFFI and ExIFFI.

## Structure

- `MyEnsemble.py` — wrappers for IF, EIF, and DIF models
- `MyInterpreter.py` — GFI/LFI computation for DIFFI, ExIFFI, KNNIFE (`kNN`), and sKNNIFE (`d`)
- `diffi_interpretability_module.py` — DIFFI baseline implementation
- `my_datasets/` — benchmark datasets
- `performance/performance.ipynb` — anomaly detection performance (ROC-AUC, AP, etc.) for IF/EIF/DIF
- `interpretability_evaluation/GFI_evaluation.ipynb` — Global Feature Importance and AUC-FS evaluation for all methods

