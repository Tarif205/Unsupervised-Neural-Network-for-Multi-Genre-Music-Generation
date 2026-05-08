# Unsupervised Neural Network for Multi-Genre Music Generation

Course: **CSE425 / EEE474 Neural Networks**

This project implements **unsupervised deep learning models for music generation using MIDI datasets**. 
The system learns musical structure such as melody, rhythm, and harmony, and generates new music sequences using various architectures including LSTM Autoencoders, Variational Autoencoders (VAE), and Transformers.

---

## MIDI Files (Google Drive)

The generated MIDI files and model checkpoints can be found at the following link:
[MIDI Drive Link](https://drive.google.com/drive/folders/1Xh4Pe0oVxH9r1ffkLABWkQvUrdw80F2R?usp=sharing)

---

## Group Contributions

| Name | Contribution |
| :--- | :--- |
| **Nasrul Azam Raf** | EDA + Preprocessing |
| **Asif Islam** | Task 1 (Autoencoder) and Task 2 (VAE) |
| **Kazi Tarif Rahman** | Task 3 (Transformer) |

---

## File Overview

### Notebooks
- `Task1_LSTM_Autoencoder.ipynb`: Implementation, training, and evaluation of the Task 1 LSTM Autoencoder.
- `Task2_VAE.ipynb`: Implementation, training, and evaluation of the Task 2 Variational Autoencoder.
- `Task3_Transformer.ipynb`: Implementation, training, and evaluation of the Task 3 Transformer-based generator.
- `preprocessing.ipynb`: Exploratory Data Analysis and data preparation.
- `baseline_markov.ipynb`: Markov Chain baseline implementation for comparison.

### Source Code (`src/`)
- **`models/`**: Core model architectures.
  - `autoencoder.py`: LSTM Autoencoder model.
  - `vae.py`: Variational Autoencoder model.
  - `transformer.py`: GPT-style Transformer Decoder model.
- **`training/`**: Training scripts for each task.
  - `train_ae_v3.py`: Training script for Task 1.
  - `train_vae.py`: Training script for Task 2.
  - `train_transformer.py`: Training script for Task 3.
- **`generation/`**: Scripts for generating MIDI samples from trained models.
  - `generate_task1_reconstruction.py`: Task 1 generation.
  - `generate_task2_samples.py`: Task 2 generation.
  - `generate_task3.py`: Task 3 generation.
  - `midi_export.py`: Utilities for converting model outputs back to MIDI files.
- **`preprocessing/`**: Data parsing and tokenization.
  - `midi_parser.py`: Parses raw MIDI files into numerical representations.
  - `tokenize_midi.py`: Converts MIDI files into REMI token sequences for the Transformer.
- **`evaluation/`**: Metric calculation.
  - `metrics.py`: Unified evaluation framework.
  - `pitch_histogram.py`: Pitch distribution analysis.
  - `rhythm_score.py`: Rhythm diversity and repetition metrics.
- `config.py`: Global configuration and hyperparameters.
- `dataset.py`: PyTorch Dataset implementation for piano rolls.

---

## Requirements

The project uses **Python 3.9+**.

Main libraries:
* PyTorch
* NumPy
* PrettyMIDI
* Miditok
* Matplotlib
* Scikit-learn
* Jupyter Notebook

All dependencies are listed in `requirements.txt`.

---

## Installing Dependencies

### 1. Create a Virtual Environment (Recommended)

#### Windows (PowerShell)
```powershell
python -m venv venv
venv\Scripts\activate
```

#### Linux / macOS
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Libraries
```bash
pip install -r requirements.txt
```

---

## Running the Project

### Step 1 — Place MIDI Dataset
Put your MIDI files inside `data/raw_midi/`.

### Step 2 — Run Preprocessing
For Task 1 & 2:
```bash
python src/preprocessing/midi_parser.py
```
For Task 3:
```bash
python src/preprocessing/tokenize_midi.py
```

### Step 3 — Train and Generate
You can run the training and generation scripts directly or use the provided Jupyter Notebooks for a more interactive experience.

Example for Task 3:
```bash
python src/training/train_transformer.py
python src/generation/generate_task3.py
```

Generated MIDI files will be saved in `outputs/generated_midis/`.

---

## Evaluation
Evaluation metrics include:
* **Pitch Histogram Similarity**: Measures how well the model captures the pitch distribution.
* **Rhythm Diversity Score**: Evaluates the variety of rhythmic patterns.
* **Repetition Ratio**: Measures the amount of repetition in the generated sequences.

Evaluation scripts are located in `src/evaluation/`.
