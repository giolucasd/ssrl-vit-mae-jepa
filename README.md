# Self-Supervised Representation Learning with ViT, MAE, and JEPA <!-- omit from toc -->

![GitHub repo size](https://img.shields.io/github/repo-size/giolucasd/ssrl-vit-mae-jepa)
![GitHub contributors](https://img.shields.io/github/contributors/giolucasd/ssrl-vit-mae-jepa)
![GitHub stars](https://img.shields.io/github/stars/giolucasd/ssrl-vit-mae-jepa?style=social)
![GitHub forks](https://img.shields.io/github/forks/giolucasd/ssrl-vit-mae-jepa?style=social)

This project investigates **self-supervised representation learning (SSRL)** for image classification using the **STL-10 dataset**.
It compares two paradigms — **Masked Autoencoders (MAE)** and **Joint Embedding Predictive Architectures (JEPA)** — with **Vision Transformer (ViT)** backbones to evaluate how pre-training on unlabeled data affects downstream classification performance.

The experiments aim to understand how increasing exposure to unlabeled data improves generalization while minimizing the need for labeled examples.

- [1. Prerequisites](#1-prerequisites)
- [2. Installing `ssrl-vit-mae-jepa`](#2-installing-ssrl-vit-mae-jepa)
- [3. Using `ssrl-vit-mae-jepa`](#3-using-ssrl-vit-mae-jepa)
  - [3.1. 🗂 Directory Structure (Scripts)](#31--directory-structure-scripts)
  - [3.2. ⚙️ Configuration (configs/mae.yaml)](#32-️-configuration-configsmaeyaml)
  - [3.3. 🧠 MAE Pre-Training (scripts.training.pretrain\_mae)](#33--mae-pre-training-scriptstrainingpretrain_mae)
  - [3.4. 🧮 Supervised Training / Fine-Tuning (scripts.training.train\_mae)](#34--supervised-training--fine-tuning-scriptstrainingtrain_mae)
    - [3.4.1. 🧊 Frozen Encoder Training (Linear-style probe)](#341--frozen-encoder-training-linear-style-probe)
    - [3.4.2. 🔥 Fine-Tuning the Encoder](#342--fine-tuning-the-encoder)
  - [3.5. 🧪 Evaluation (scripts.evaluation.evaluate\_classifier)](#35--evaluation-scriptsevaluationevaluate_classifier)
  - [3.6. 🔍 Representation Visualization (scripts.evaluation.visualize\_representation)](#36--representation-visualization-scriptsevaluationvisualize_representation)
  - [3.7. 🔬 Full Ablation Studies](#37--full-ablation-studies)
    - [3.7.1. 🧩 Pre-training Ablation](#371--pre-training-ablation)
    - [3.7.2. 🧠 Downstream Training Ablation](#372--downstream-training-ablation)
    - [3.7.3. Summary](#373-summary)
- [4. Contributing](#4-contributing)
- [5. Contributors](#5-contributors)
- [6. Contact](#6-contact)
- [7. License](#7-license)

---

## 1. Prerequisites

Before you begin, ensure you have met the following requirements:

* You have **[uv](https://github.com/astral-sh/uv)** installed (for dependency management and reproducibility).
* You have **Python 3.13+** installed.
* The project was tested on **Linux**.
* A CUDA-enabled GPU is strongly recommended for efficient pre-training and fine-tuning.

---

## 2. Installing `ssrl-vit-mae-jepa`

Clone this repository and install dependencies using **uv**:

```bash
git clone https://github.com/giolucasd/ssrl-vit-mae-jepa.git
cd ssrl-vit-mae-jepa
uv sync
```

To include development dependencies (for reproducibility or debugging):

```bash
uv sync --all-extras
```

To check if the installation was succesful and GPU access is correct:

```bash
uv run python tests/test_cuda_torch.py
uv run python tests/test_cuda_benchmark.py
```

After installing the dependencies, activate the virtual environment created by **uv**:

```bash
source .venv/bin/activate
```

---

## 3. Using `ssrl-vit-mae-jepa`

The project provides a modular and fully script-driven pipeline for **self-supervised pre-training**, **downstream fine-tuning**, **evaluation**, and **ablation studies**.

All scripts include built-in help:
```bash
python -m scripts.training.pretrain_mae --help
python -m scripts.training.train_mae --help
python -m scripts.evaluation.evaluate_classifier --help
python -m scripts.evaluation.visualize_representation --help
```

### 3.1. 🗂 Directory Structure (Scripts)

```bash
scripts/
 ├─ training/
 │   ├─ pretrain_mae.py
 │   └─ train_mae.py
 │
 ├─ evaluation/
 │   ├─ evaluate_classifier.py
 │   ├─ visualize_representation.py
 │   └─ visualize_reconstruction.py
 │
 ├─ ablation/
 │   ├─ run_pretrain_ablation.py
 │   └─ run_train_ablation.py
 │
 ├─ data.py
 └─ utils.py
```

### 3.2. ⚙️ Configuration (configs/mae.yaml)

All training scripts use the same unified YAML configuration:

```yaml
model:
  general:
    image_size: 96
    patch_size: 8
    in_chans: 3

  encoder:
    embed_dim: 144
    depth: 4
    num_heads: 6

  decoder:
    decoder_embed_dim: 192
    decoder_depth: 2
    decoder_num_heads: 6

  head:
    embed_dim: 144
    pool: cls

pretrain:
  mask_ratio_start: 0.75
  mask_ratio_end: 0.75
  mask_ramp_epochs: 5
  total_epochs: 800
  warmup_epochs: 20
  batch_size: 2000
  base_learning_rate: 1.5e-4
  weight_decay: 0.05
  data_fraction: 1.00
  val_split: 0.06
  num_workers: 4

train:
  samples_per_class: 400
  total_epochs: 100
  warmup_epochs: 10
  batch_size: 2000
  learning_rate: 3e-4
  weight_decay: 0.05
  freeze_encoder: true
  num_workers: 4

test:
  batch_size: 2000
  num_workers: 4

logging:
  output_dir_base: outputs
  model_path: vit-mae.pt
```

### 3.3. 🧠 MAE Pre-Training (scripts.training.pretrain_mae)

Runs Masked Autoencoder pre-training on unlabeled STL-10.

```bash
python -m scripts.training.pretrain_mae \
  --config configs/mae.yaml \
  --output_dir_suffix mae_100
```

Outputs are stored in:

```bash
outputs/pretrain/<suffix>/
    checkpoints/
        best.ckpt
        last.ckpt
    logs/
    vit-mae.pt
```

Use --output_dir_suffix to specify runs like mae_025, mae_050, mae_075, mae_100. Note that `scripts/ablation/run_pretrain_ablation.py` does that automatically for all predefined ablation percentages.

### 3.4. 🧮 Supervised Training / Fine-Tuning (scripts.training.train_mae)
#### 3.4.1. 🧊 Frozen Encoder Training (Linear-style probe)

```bash
python -m scripts.training.train_mae \
  --config configs/mae.yaml \
  --encoder_ckpt outputs/pretrain/mae_100/checkpoints/best.ckpt \
  --output_dir_suffix mae_100_400
```

Trains only the classification head while keeping encoder frozen.

#### 3.4.2. 🔥 Fine-Tuning the Encoder

Continue from a frozen head checkpoint:

```bash
python -m scripts.training.train_mae \
  --config configs/mae.yaml \
  --classifier_ckpt outputs/train/mae_100_400/checkpoints/best.ckpt \
  --output_dir_suffix mae_100_400_finetuned
```

If freeze_encoder: false is set in the config, the full encoder is unfrozen (or partially unfrozen when using unfreeze_last_layers).

### 3.5. 🧪 Evaluation (scripts.evaluation.evaluate_classifier)

Evaluate any trained classifier checkpoint on the STL-10 test split:

```bash
python -m scripts.evaluation.evaluate_classifier \
  --config configs/mae.yaml \
  --checkpoint outputs/train/mae_100_400_finetuned/checkpoints/best.ckpt
```

Outputs:
- test accuracy
- logs under outputs/test/\<suffix\>/logs/
- reusable evaluation function within script

### 3.6. 🔍 Representation Visualization (scripts.evaluation.visualize_representation)

Generates t-SNE or UMAP visualizations of encoder features.

```bash
python -m scripts.evaluation.visualize_representation \
  --config configs/mae.yaml \
  --encoder_ckpt outputs/pretrain/mae_100/vit-mae.pt \
  --method umap \
  --pool cls \
  --normalize none
```

Saves images to `assets/visualizations/representation_*.png`.

Supports:
- pooling: cls, mean
- normalization: none, l2, channel
- UMAP (recommended) or t-SNE

### 3.7. 🔬 Full Ablation Studies

Two scripts automatically run the entire set of experiments.

#### 3.7.1. 🧩 Pre-training Ablation

Trains 4 models with different fractions of unlabeled data:

```bash
python -m scripts.ablation.run_pretrain_ablation
```

Runs sequential pre-training for:
- 25%
- 50%
- 75%
- 100%


#### 3.7.2. 🧠 Downstream Training Ablation

Runs 112 experiments covering:
- 4 unlabeled data fractions ×
- 7 label budgets (10–400/class) ×
- 4 training modes (frozen, +1 layer, +2 layers, full fine-tune)

```bash
python -m scripts.ablation.run_train_ablation
```

This script:
- Automatically loads correct checkpoints at each stage
- Runs frozen training → unfreeze-1 → unfreeze-2 → full FT
- Stores results in:

```bash
outputs/train/<frac>_<labels>_frozen/
outputs/train/<frac>_<labels>_unfreeze1/
outputs/train/<frac>_<labels>_unfreeze2/
outputs/train/<frac>_<labels>_full/
```

#### 3.7.3. Summary

The workflow is:

1. Run pretraining ablation
2. Run downstream ablation
3. Optional:
   1. Evaluate final classifiers
   2. Visualize learned representations

All results are reproducible, saved within outputs/, and linked to their original configuration snapshot.

---

## 4. Contributing

To contribute to this project:

1. Fork this repository.
2. Create a feature branch:

   ```bash
   git checkout -b feature/<branch_name>
   ```
3. Commit your changes:

   ```bash
   git commit -m "feat: <description>"
   ```
4. Push to your fork and open a Pull Request.

---

## 5. Contributors

* [@giolucasd](https://github.com/giolucasd)
* [@yanprada](https://github.com/yanprada)

---

## 6. Contact

For questions or collaboration, reach out to:

📧 [g173317@dac.unicamp.br](mailto:g173317@dac.unicamp.br)
📧 [y118982@dac.unicamp.br](mailto:y118982@dac.unicamp.br)

---

## 7. License

This project is licensed under the [MIT License](LICENSE).

---
