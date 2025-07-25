# Transformer Translation

A PyTorch-based sequence-to-sequence Transformer model for text translation, with a focus on ancient Chinese language processing.

---

## Badges

<!-- Add badges here if you use CI, coverage, etc. Example: -->
<!-- ![Build Status](https://img.shields.io/github/workflow/status/yourusername/yourrepo/CI) -->
<!-- ![License](https://img.shields.io/github/license/yourusername/yourrepo) -->

---

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Screenshots / Demo](#screenshots--demo)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

---

## Introduction

**Transformer Translation** is a deep learning project implementing a Transformer-based model for text translation, particularly suited for ancient Chinese. It provides scripts for data preprocessing, training, and evaluation, and is designed for researchers, students, and developers interested in NLP and machine translation.

**Main Features:**
- Customizable Transformer architecture (layers, heads, etc.)
- Data preprocessing utilities for parallel corpora
- Training and evaluation scripts
- Tokenization using pretrained BERT for ancient Chinese ([Jihuai/bert-ancient-chinese](https://huggingface.co/Jihuai/bert-ancient-chinese))
- Modular and extensible codebase

---

## Project Structure

```
transformer-translation/
├── configs/           # Configuration files for experiments
├── checkpoints/       # Saved model checkpoints
├── data/              # Data files (raw, processed, train/val/test splits)
├── logs/              # Training and evaluation logs
├── model/             # Model architecture (Transformer, attention, etc.)
├── scripts/           # Training, evaluation, and config scripts
├── utils/             # Utilities: dataloader, tokenizer, helpers
├── requirements.txt   # Python dependencies
├── README.md          # Project documentation
└── .gitignore         # Git ignore rules
```

- **configs/**: Store experiment and model configuration files.
- **checkpoints/**: Contains saved model weights after training.
- **data/**: Place your dataset files here. Preprocessing scripts will generate train/val/test splits in this folder.
- **logs/**: Output logs from training and evaluation runs.
- **model/**: All model-related code, including the Transformer architecture and attention mechanisms.
- **scripts/**: Main scripts for training (`train.py`), evaluation (`evaluate.py`), and configuration (`train_config.py`).
- **utils/**: Helper modules for data loading, tokenization, and preprocessing.
- **requirements.txt**: List of required Python packages.
- **README.md**: This documentation file.
- **.gitignore**: Specifies files and directories to be ignored by git.

---

## Data and Tokenizer Sources

- **Dataset**: The dataset comes from [ClassicalModernCorpus](https://github.com/Hellohistory/ClassicalModernCorpus), a cleaned and curated collection of 文白对照语料, originally sourced from:
  - [BangBOOM/Classical-Chinese](https://github.com/BangBOOM/Classical-Chinese)
  - [NiuTrans/Classical-Modern](https://github.com/NiuTrans/Classical-Modern)
- **Tokenizer**: [Jihuai/bert-ancient-chinese](https://huggingface.co/Jihuai/bert-ancient-chinese) — a BERT model pretrained for ancient Chinese, with an expanded vocabulary and domain-adaptive pretraining. For more details, see the [model card and documentation](https://huggingface.co/Jihuai/bert-ancient-chinese).

---

## Screenshots / Demo

<!-- Add screenshots or demo GIFs here if available -->
<!-- Example: -->
<!-- ![Demo](docs/demo.gif) -->

---

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- `transformers` library (for tokenizer)
- See `requirements.txt` for full dependencies

### Installation

```bash
git clone https://github.com/GG772/transformer-translation.git
cd transformer-translation
pip install -r requirements.txt
```

---

## Usage

### Data Preparation

1. Unzip data in the `data/` directory.
2. Use the helper script to preprocess data:
   ```bash
   python utils/helpers.py
   ```
   This will generate `train_src.txt`, `train_tgt.txt`, `val_src.txt`, and `val_tgt.txt` in `data/`.

### Training

```bash
python scripts/train.py
```
- Model checkpoints are saved in `checkpoints/`.

### Evaluation

```bash
python scripts/evaluate.py
```
- Prints sample translations and model predictions.

---

## Configuration

Edit `scripts/train_config.py` to adjust:
- Number of epochs
- Learning rate
- Batch size
- Model dimensions (d_model, n_head, etc.)

---

## API Reference

- `model/model.py`: Transformer, Encoder, Decoder, and attention modules
- `utils/dataloader.py`: CustomDataset for loading and batching data
- `utils/tokenizer.py`: Loads and configures the BERT tokenizer for ancient Chinese

---

## Examples

- See `scripts/train.py` and `scripts/evaluate.py` for end-to-end training and evaluation workflows.

---

## Contributing

Contributions are welcome! Please open issues or pull requests.

- For major changes, open an issue first to discuss your ideas.
- Please make sure to update tests as appropriate.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [Jihuai/bert-ancient-chinese](https://huggingface.co/Jihuai/bert-ancient-chinese)
- Inspiration from the original Transformer paper: ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762)

---

## Contact

Maintainer: [George]  
Email: [whgeorgewu5@gmail.com]

---
