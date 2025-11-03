# Self-Supervised Representation Learning with ViT, MAE, and JEPA

![GitHub repo size](https://img.shields.io/github/repo-size/giolucasd/ssrl-vit-mae-jepa)
![GitHub contributors](https://img.shields.io/github/contributors/giolucasd/ssrl-vit-mae-jepa)
![GitHub stars](https://img.shields.io/github/stars/giolucasd/ssrl-vit-mae-jepa?style=social)
![GitHub forks](https://img.shields.io/github/forks/giolucasd/ssrl-vit-mae-jepa?style=social)

This project investigates **self-supervised representation learning (SSRL)** for image classification using the **STL-10 dataset**.
It compares two paradigms — **Masked Autoencoders (MAE)** and **Joint Embedding Predictive Architectures (JEPA)** — with **Vision Transformer (ViT)** backbones to evaluate how pre-training on unlabeled data affects downstream classification performance.

The experiments aim to understand how increasing exposure to unlabeled data improves generalization while minimizing the need for labeled examples.

---

## Prerequisites

Before you begin, ensure you have met the following requirements:

* You have **Python 3.11+** installed.
* You have **[uv](https://github.com/astral-sh/uv)** installed (for dependency management and reproducibility).
* The project was tested on **Linux**.
* A CUDA-enabled GPU is strongly recommended for efficient pre-training and fine-tuning.

---

## Installing `ssrl-vit-mae-jepa`

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

---

## Using `ssrl-vit-mae-jepa`

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

## Contributing

To contribute to this project:

1. Fork this repository.
2. Create a feature branch:

   ```bash
   git checkout -b feature/<branch_name>
   ```
3. Commit your changes:

   ```bash
   git commit -m "Add <description>"
   ```
4. Push to your fork and open a Pull Request.

---

## Contributors

* [@giolucasd](https://github.com/giolucasd)
* [@yanprada](https://github.com/yanprada)

---

## Contact

For questions or collaboration, reach out to:

📧 [g173317@dac.unicamp.br](mailto:g173317@dac.unicamp.br)

---

## License

This project is licensed under the [MIT License](LICENSE).

---
