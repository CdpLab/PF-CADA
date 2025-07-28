# PF‑CADA: Multi‑stage Pyramid Feature Fusion & Capsule‑enhanced Adversarial Domain Adaptation for EEG Emotion Recognition

**PF‑CADA** is a deep model for **cross‑subject EEG emotion recognition** that integrates:

* **Multi‑stage Feature Pyramid (FPN)** — fuses spatial/frequency multi‑scale semantics, preserving both global context and local details.
* **Capsule Network with Dynamic Routing** — models hierarchical relations and fine‑grained differences.
* **Adversarial Domain Adaptation (GRL)** — aligns source/target feature distributions and mitigates negative transfer.
* **Mamba‑style Selective State‑Space Classifier** — performs selective encoding and recursive state updates for long EEG sequences.

This repo provides complete scripts for **DEAP** and **SEED** under both **subject‑dependent** and **subject‑independent** protocols.

---

![Uploading image.png…]()


## Repository Structure

```text
.
├─ pf_cada_model.py                # Core model: FPN + Capsules + GRL + Mamba-style classifier
├─ deap_dependent.py               # DEAP: subject-dependent protocol
├─ deap_independent.py             # DEAP: subject-independent (adversarial DA + finetune)
├─ seed_dependent.py               # SEED: subject-dependent protocol
├─ seed_independent.py             # SEED: subject-independent (adversarial DA + finetune)
└─ exercise/                       # Results & confusion matrices (created after runs)
```

---

## Requirements

* Python ≥ 3.7
* PyTorch ≥ 1.10 (CUDA strongly recommended)
* NumPy, SciPy, scikit‑learn

---

## Data Preparation

> Update paths at the top of each script to match your local setup.

### DEAP

* The scripts expect preprocessed Differential Entropy (DE) features. Directory example:

  ```
  D:/DEAP/with_base_0.5/
    ├── DE_s01
    ├── DE_s02
    └── ... DE_s32
  ```
* Each `DE_sXX` is a `.mat` file containing:

  * `data`  (shape like: trial × band × H × W)
  * `valence_labels`, `arousal_labels` (binary high/low)

Set in `deap_*.py`:

```python
dataset_dir = "D:/DEAP/with_base_0.5/"
flag = 'v'  # 'v' for Valence, 'a' for Arousal
```

### SEED

* The scripts expect `.npy` files (example):

  ```
  D:/SEED/SEEDzyr/DE0.5s/t6x_89.npy
  D:/SEED/SEEDzyr/DE0.5s/t6y_89.npy
  ```

## Key Hyperparameters & Tips

* **Domain loss weight** `lambda_d`: try `0.05 ~ 0.5` (default `0.1`).
* **Primary capsules** `primary_caps_maps`: default `32`; reduce to `16` if VRAM is tight.
* **Batch size**: default `256`; tune based on memory.
* **Optimizer**: default `SGD(lr=0.01, momentum=0.9, weight_decay=1e-4)`; AdamW is also viable.
* **Mamba classifier hidden dim**: default `256` (`MambaClassifier(..., hidden=256, ...)`).
* **Random seed**: default `3407` for reproducibility.
* **Task switch (DEAP)**: set `flag='v'` (Valence) or `'a'` (Arousal).

---

## Reported Results (from the Paper)

> Representative numbers (subject‑independent emphasized):

* **SEED (subject‑independent)**: **91.93%**
* **DEAP (subject‑independent)**: Valence **71.92%** / Arousal **73.38%**
* **DEAP (subject‑dependent)**: Valence **96.71%** / Arousal **97.69%**
* **SEED (subject‑dependent)**: **98.11%**
* **User study (real‑world)**: up to **72.80%**

---

## License & Acknowledgments

* The code is provided for **research purposes only**. Please also comply with the usage licenses of the **DEAP** and **SEED** datasets.
* We thank the dataset creators and the open‑source community.

---

## Contact

Questions, reproducibility, or feature requests (multi‑GPU, CLI arguments, logging/visualization, Docker, etc.) — feel free to open an issue or contact us.

