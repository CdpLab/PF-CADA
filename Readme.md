Below is an **English README.md** you can place at the root of your repository. It reflects your paper and the current code layout (`pf_cada_model.py` + four experiment scripts).

---

# PF‑CADA: Multi‑stage Pyramid Feature Fusion & Capsule‑enhanced Adversarial Domain Adaptation for EEG Emotion Recognition

**PF‑CADA** is a deep model for **cross‑subject EEG emotion recognition** that integrates:

* **Multi‑stage Feature Pyramid (FPN)** — fuses spatial/frequency multi‑scale semantics, preserving both global context and local details.
* **Capsule Network with Dynamic Routing** — models hierarchical relations and fine‑grained differences.
* **Adversarial Domain Adaptation (GRL)** — aligns source/target feature distributions and mitigates negative transfer.
* **Mamba‑style Selective State‑Space Classifier** — performs selective encoding and recursive state updates for long EEG sequences.

This repo provides complete scripts for **DEAP** and **SEED** under both **subject‑dependent** and **subject‑independent** protocols.

---

## Highlights

* **End‑to‑end cross‑subject pipeline** with adversarial domain alignment.
* **Decoupled design**: all model components live in `pf_cada_model.py`; experiment scripts only **import and call** the model (no duplicated networks).
* **Robust to sequence length** with a selective state‑space (Mamba‑style) classifier.
* **Reproducible splits** and simple configuration via script constants.

---

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

* Python ≥ 3.8
* PyTorch ≥ 1.10 (CUDA strongly recommended)
* NumPy, SciPy, scikit‑learn

Install example:

```bash
# Pick the right CUDA build for your system if you have a GPU:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install numpy scipy scikit-learn
```

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
* `t6x_89.npy`: shape like (subject\*session, T=6, H=8, W=9, C=5). Scripts select 4 frequency bands `[1:5]`.
* `t6y_89.npy`: label indices.

Adjust the real paths in `seed_*.py`.

---

## Quick Start

> GPU is highly recommended (the scripts auto‑detect `cuda`).

### DEAP — Subject‑dependent

```bash
python deap_dependent.py
```

Flow:

1. Randomly split **20%** for pre‑training, **80%** for 5‑fold finetune/test.
2. Use `MyModel` (from `pf_cada_model.py`) for classification **without** domain alignment.
3. Average accuracy, F1, and confusion matrix are saved in `exercise/`.

### DEAP — Subject‑independent

```bash
python deap_independent.py
```

Flow:

1. **Subject 0**: take **\~20%** as target finetune set, the rest as target test set.
2. **Other 31 subjects** are the **source domain**: run 5‑fold **adversarial domain adaptation** (classification loss + domain loss).
3. Finetune on the target subset, then evaluate on subject‑0 test set. Results saved in `exercise/`.

### SEED — Subject‑dependent

```bash
python seed_dependent.py
```

Flow:

1. Random **80/20** split after a small pre‑training phase.
2. 5‑fold finetune + test, save average accuracy/F1/confusion matrix.

### SEED — Subject‑independent

```bash
python seed_independent.py
```

Flow:

1. **Subject 0** is target domain (default **10%** finetune, rest test — configurable in the script).
2. **Other 14 subjects** are source domain for 5‑fold **adversarial DA**.
3. Finetune on the target subset, then test on subject‑0 test set.

---

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

> Your local results may vary due to preprocessing, random splits, hardware, and hyperparameters.

---

## Troubleshooting

1. **Out‑of‑Memory (OOM)**

   * Decrease `batchsize`; reduce `primary_caps_maps` (e.g., 32 → 16); consider smaller hidden sizes.

2. **Unstable training / slow convergence**

   * Adjust `lambda_d`; ramp GRL strength over epochs; delay DA start; increase pre‑train/finetune epochs.

3. **Path errors**

   * Double‑check `dataset_dir` (DEAP) and `.npy` paths (SEED) and update them at the top of the scripts.

4. **Sequence length mismatch**

   * The code assumes `t_steps=6` (0.5 s windows per sample). If you change the windowing, ensure data and scripts agree.

---

## Citation

If you find this work useful, please cite our paper:

```
Jia Liu, Yangrui Zhang, Chengcheng Hua, Lina Wei, and Dapeng Chen,
"PF-CADA: A Multi-stage Pyramid Feature Fusion and Capsule-enhanced Adversarial Domain Adaptation for EEG Emotion Recognition,"
IEEE Sensors Journal, 2025.
```

BibTeX:

```bibtex
@article{liu2025pfcada,
  title   = {PF-CADA: A Multi-stage Pyramid Feature Fusion and Capsule-enhanced Adversarial Domain Adaptation for EEG Emotion Recognition},
  author  = {Liu, Jia and Zhang, Yangrui and Hua, Chengcheng and Wei, Lina and Chen, Dapeng},
  journal = {IEEE Sensors Journal},
  year    = {2025}
}
```

---

## License & Acknowledgments

* The code is provided for **research purposes only**. Please also comply with the usage licenses of the **DEAP** and **SEED** datasets.
* We thank the dataset creators and the open‑source community.

---

## Contact

Questions, reproducibility, or feature requests (multi‑GPU, CLI arguments, logging/visualization, Docker, etc.) — feel free to open an issue or contact us.
If you want a further refactor (e.g., a shared `train_utils.py` and command‑line flags like `--dataset`, `--protocol`, `--lambda_d`), we can add it on top of the current repo structure.

---

*Happy research!* 🎓🧠💡
