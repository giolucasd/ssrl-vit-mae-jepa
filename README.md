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
- [4. Contributing](#4-contributing)
- [5. Contributors](#5-contributors)
- [6. Contact](#6-contact)
- [7. License](#7-license)

---

## 1. Prerequisites

Before you begin, ensure you have met the following requirements:

* You have **Python 3.11+** installed.
* You have **[uv](https://github.com/astral-sh/uv)** installed (for dependency management and reproducibility).
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
uv run python test/test_cuda_torch.py
uv run python test/test_cuda_benchmark.py
```

---

## 3. Using `ssrl-vit-mae-jepa`

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
