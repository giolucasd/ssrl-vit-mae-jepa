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
  - [🧩 `scripts/data.py`](#-scriptsdatapy)
  - [🧠 `scripts/pretrain_mae.py`](#-scriptspretrain_maepy)
  - [🧮 `scripts/train_mae.py`](#-scriptstrain_maepy)
  - [🧪 `scripts.linear_probe.py`](#-scriptslinear_probepy)
  - [📊 `scripts.evaluate_classifier.py`](#-scriptsevaluate_classifierpy)
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

The project provides a set of modular training scripts that implement all stages of the self-supervised learning pipeline — from data preparation to evaluation.

---

### 🧩 `scripts/data.py`
Utility for downloading and verifying the **STL-10** dataset (labeled + unlabeled).

```bash
python -m scripts.data
```

By default, the dataset is stored under `data/`.

---

### 🧠 `scripts/pretrain_mae.py`
Runs **Masked Autoencoder (MAE)** pre-training on the unlabeled subset of STL-10.

```bash
python -m scripts.pretrain_mae \
  --data_fraction 0.25 \
  --total_epochs 50 \
  --warmup_epochs 5 \
  --batch_size 512 \
  --max_device_batch_size 512 \
  --model_path vit-mae-025.pt \
  --output_dir outputs/pretrain/mae_025
```

**Key arguments:**
- `--data_fraction`: fraction of unlabeled STL-10 data used for pre-training (`0.25`, `0.5`, `1.0`, etc.).
- `--total_epochs`: total number of training epochs.
- `--warmup_epochs`: linear warmup duration before applying the cosine scheduler.
- `--batch_size`: global batch size.
- `--output_dir`: directory to save checkpoints and logs.

---

### 🧮 `scripts/train_mae.py`
Fine-tunes the pretrained encoder on labeled STL-10 samples (either frozen or unfrozen).

```bash
python -m scripts.train_mae \
  --encoder_ckpt outputs/pretrain/mae_025/checkpoints/last.ckpt \
  --freeze_encoder True \
  --samples_per_class 400 \
  --epochs 100 \
  --lr 3e-4 \
  --output_dir outputs/train/mae_400
```

To **fine-tune the entire encoder** (unfrozen):
```bash
python -m scripts.train_mae \
  --classifier_ckpt outputs/train/mae_400/checkpoints/best-valacc-epoch=078-val_acc=0.3080.ckpt \
  --freeze_encoder False \
  --epochs 50 \
  --lr 1e-5 \
  --output_dir outputs/train/mae_400_finetune
```

**Key arguments:**
- `--encoder_ckpt`: path to pretrained MAE encoder checkpoint.
- `--classifier_ckpt`: path to previously trained classifier (optional, for fine-tuning).
- `--freeze_encoder`: whether to freeze encoder weights.
- `--samples_per_class`: number of labeled examples per class (10–400).
- `--epochs`: total number of fine-tuning epochs.

---

### 🧪 `scripts.linear_probe.py`
Evaluates **frozen encoder representations** by training a single linear classifier on top of them (linear probe).

```bash
python -m scripts.linear_probe \
  --encoder_ckpt outputs/pretrain/mae_100/checkpoints/mae-epoch=394-train_loss=0.062.ckpt \
  --batch_size 1024 \
  --epochs 50 \
  --lr 1e-3 \
  --output_dir outputs/linear_probe
```

**Output:** linear probe accuracy (top-1) on STL-10 test set.

---

### 📊 `scripts.evaluate_classifier.py`
Evaluates a fine-tuned classifier checkpoint on the STL-10 **test set**.

```bash
python -m scripts.evaluate_classifier \
  --ckpt_path outputs/train/mae_400_finetune/checkpoints/best-valacc-epoch=020-val_acc=0.3110.ckpt \
  --batch_size 256 \
  --num_workers 8
```

**Output:**
- Accuracy, precision, recall, and F1-score metrics on the test set.

---

All results — including model checkpoints, logs, and TensorBoard summaries — are saved in the `outputs/` directory.

---

Train a model using MAE or JEPA pre-training followed by fine-tuning:

```bash
uv run python scripts/train.py --method mae --pretrain 0.5 --finetune 100
```

Example options:

* `--method {mae,jepa}` — self-supervised paradigm.
* `--pretrain {0.25,0.5,0.75,1.0}` — fraction of unlabeled data used for pre-training.
* `--finetune {10,25,50,100,200,300,400}` — number of labeled examples per class for fine-tuning.

Results (checkpoints, logs, and metrics) are saved under the `outputs/` directory.
A reproducible **Jupyter notebook** for visualizing results and attention heatmaps is included in `notebooks/analysis.ipynb`.

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
