# NMT-Transformer

This repository contains an implementation of a **Transformer model** for **Neural Machine Translation (NMT)**, designed to translate text from **English to Russian**. The project uses **TensorFlow** for the model implementation, **SentencePiece** for subword tokenization, and **PyGAD** for hyperparameter optimization via a genetic algorithm. It also includes a **Flask-based web interface** for interactive translation.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Hyperparameter Optimization](#hyperparameter-optimization)
  - [Running the Web Interface](#running-the-web-interface)
  - [Inference in Jupyter Notebook](#inference-in-jupyter-notebook)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview
The `NMT-Transformer` project implements a Transformer-based model for translating English sentences to Russian, based on the architecture introduced in *Attention Is All You Need* (Vaswani et al., 2017). The codebase includes scripts for data preprocessing, model training, hyperparameter optimization, and a web interface for real-time translation. The model is trained on a dataset of English-Russian sentence pairs and uses subword tokenization to handle diverse vocabularies efficiently.

## Features
- **Transformer Model**: Encoder-decoder architecture with multi-head self-attention, positional encodings, and feed-forward networks.
- **Subword Tokenization**: Uses SentencePiece for efficient tokenization with a vocabulary size of 12,000.
- **Custom Loss and Metrics**: Masked sparse categorical cross-entropy and accuracy metrics to ignore padding and start tokens.
- **Hyperparameter Optimization**: Genetic algorithm (PyGAD) to optimize model hyperparameters like number of layers, model dimension, and dropout rate.
- **Web Interface**: Flask-based UI for translating English sentences to Russian with light/dark mode support.
- **Jupyter Notebook**: Example notebook for loading the model and performing translations.

## Requirements
- Python 3.8+
- TensorFlow 2.x
- SentencePiece
- PyGAD
- Flask
- NumPy
- pandas (for dataset loading)
- A GPU is recommended for faster training.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/NikitaGoldashevsky/NMT-Transformer.git
   cd NMT-Transformer
   ```
2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install tensorflow sentencepiece pygad flask numpy pandas
   ```
4. Download or prepare the dataset (`rus_300000.csv`) and place it in the project root.

## Dataset
The project uses a dataset of English-Russian sentence pairs stored in `rus_300000.csv`. The file is expected to have two or three columns (e.g., English and Russian sentences, with an optional first column). The dataset is processed to create subword vocabularies using SentencePiece, with a maximum sequence length of 25 tokens.

To use your own dataset:
1. Prepare a CSV file with English and Russian sentence pairs.
2. Update the `ds_name` variable in `Training_remote.py` to point to your dataset.

## Usage

### Training the Model
1. Run the training script:
   ```bash
   python Training_remote.py
   ```
   This script:
   - Loads and preprocesses the dataset.
   - Trains a SentencePiece tokenizer.
   - Builds and trains the Transformer model for 8+2 epochs.
   - Prints sample translations after each epoch using a callback.
   - Saves the trained model (`my_transformer_model.keras`) and tokenizer files (`bpe.model`, `bpe.vocab`).

### Hyperparameter Optimization
The script includes a genetic algorithm to optimize hyperparameters:
- Run the optimization section in `Training_remote.py` (requires PyGAD).
- The algorithm tests combinations of `num_layers`, `d_model`, `num_heads`, `d_ff`, `dropout_rate`, `learning_rate`, and `batch_size`.
- Results are printed, including the best hyperparameters and validation accuracy.

### Running the Web Interface
1. Ensure the trained model (`my_transformer_model_subword_bugfixed.keras`) and tokenizer (`bpe_subword_bugfixed.model`) are in the project root.
2. Run the Flask app:
   ```bash
   python Flask_Interface.py
   ```
3. Open a browser and navigate to `http://127.0.0.1:5000`.
4. Enter an English sentence and click "Translate" to see the Russian translation and inference time.

### Inference in Jupyter Notebook
1. Open `Importing_Transformer.ipynb` in Jupyter.
2. Run the cells under the "Subword tokenization" section to load the model and tokenizer.
3. Test translations with example sentences or your own inputs.

Example:
```python
print(decode_sequence("What are you going to do this morning?"))
# Output: Что вы будете делать сегодня утром?
```

## Model Architecture
The Transformer model consists of:
- **Encoder**: Processes English input with multi-head self-attention, positional encodings, and feed-forward networks.
- **Decoder**: Generates Russian output with masked self-attention, cross-attention to encoder outputs, and feed-forward networks.
- **Hyperparameters**:
  - `d_model`: 384
  - `num_heads`: 8
  - `num_layers`: 1
  - `d_ff`: 512
  - `dropout_rate`: 0.153
  - `max_len`: 25
  - `vocab_size`: 12,000
- **Custom Layers**: `PositionalEncoding` for sequence position information and `PaddingMask` for ignoring padding tokens.
- **Loss**: Masked sparse categorical cross-entropy.
- **Optimizer**: Adam with exponential decay learning rate.

## Results
The model achieves reasonable translations for short English sentences, as shown in `Importing_Transformer.ipynb`. Example translations:
- "The cat sits on the sofa." → "Кот сидень на диване."
- "She loves to play soccer." → "Она любит играть в футбол."
- "What is your favorite color?" → "Какой твой любимый цвет?"

To evaluate performance quantitatively, consider computing BLEU scores using a library like `sacrebleu`.

## Contributing
Contributions are welcome! Please:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-name`).
5. Open a pull request.

Suggestions for improvement:
- Add BLEU score evaluation.
- Support additional language pairs.
- Enhance the genetic algorithm with more generations or a larger population.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.