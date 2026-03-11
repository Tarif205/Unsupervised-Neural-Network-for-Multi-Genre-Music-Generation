# Unsupervised Neural Network for Multi-Genre Music Generation

Course: **CSE425 / EEE474 Neural Networks**

This project implements **unsupervised deep learning models for music generation using MIDI datasets**.
The system learns musical structure such as melody, rhythm, and harmony, and generates new music sequences.

---

# Project Structure

```
music-generation-unsupervised/

README.md
requirements.txt

data/
    raw_midi/
    processed/
    train_test_split/

notebooks/

src/
    config.py
    preprocessing/
    models/
    training/
    evaluation/
    generation/

outputs/
    generated_midis/
    plots/
    survey_results/

report/
```

---

# Requirements

The project uses **Python 3.9+**.

Main libraries:

* PyTorch
* NumPy
* Pandas
* PrettyMIDI
* Music21
* Matplotlib
* Scikit-learn
* Jupyter Notebook

All dependencies are listed in **requirements.txt**.

---

# Installing Dependencies

## 1. Create a Virtual Environment (Recommended)

### Windows (PowerShell)

```
python -m venv venv
```

Activate the environment:

```
venv\Scripts\activate
```

### Linux / macOS

```
python3 -m venv venv
source venv/bin/activate
```

---

## 2. Install Libraries Using requirements.txt

Run the following command inside the project directory:

```
pip install -r requirements.txt
```

This command will automatically install **all required libraries** listed in the requirements file.

---

# Running the Project

### Step 1 — Place MIDI Dataset

Put all MIDI files inside:

```
data/raw_midi/
```

Example dataset:

* Lakh MIDI Dataset
* MAESTRO Dataset

---

### Step 2 — Run Preprocessing

Preprocess the MIDI files using the preprocessing module.

Example:

```
python src/preprocessing/midi_parser.py
```

---

### Step 3 — Train the Model

Train the Autoencoder model:

```
python src/training/train_ae.py
```

Train the VAE model:

```
python src/training/train_vae.py
```

Train the Transformer model:

```
python src/training/train_transformer.py
```

---

# Generating Music

After training, generate new music samples:

```
python src/generation/generate_music.py
```

Generated MIDI files will be saved in:

```
outputs/generated_midis/
```

You can open them using any MIDI player or DAW.

---

# Evaluation

Evaluation metrics include:

* Pitch Histogram Similarity
* Rhythm Diversity Score
* Repetition Ratio
* Human Listening Score

Evaluation scripts are located in:

```
src/evaluation/
```

---

# Outputs

Generated results will be stored in:

```
outputs/generated_midis/
outputs/plots/
outputs/survey_results/
```

---

# Authors

Group Project — Neural Networks
Department of CSE
