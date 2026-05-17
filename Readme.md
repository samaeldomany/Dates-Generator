# Dates Generator - Deep Generative Models

This repository contains the solution for **DSAI 490 Assignment 2: Dates Generator**.

The objective of this project is to explore conditional deep generative models by generating valid calendar dates (`dd-mm-yyyy`) based on a specific set of text conditions.

Given an input like:

```text
[WED] [JAN] [False] [180]
```

the models attempt to generate a date that:
- Falls on a Wednesday
- Is in January
- Is in a non-leap year
- Belongs to the 1800s decade

---

# 📂 Repository Structure

```text
repo/
│
├── data/
│   ├── data.txt                # Full dataset (conditions + target dates)
│   ├── example_input.txt       # Example conditions for inference testing
│   └── output_*.txt            # Generated outputs from the models
│
├── model/
│   ├── architectures.py        # PyTorch model definitions (VAE, GAN, Transformer, MLP)
│   ├── preprocessing.py        # Tokenization, Dataset class, and ConditionValidator
│   ├── train.py                # Training loops for all four models
│   ├── predict.py              # CLI inference script
│   └── *.pt                    # Saved model weights
│
├── environment.yml             # Conda environment specifications
└── Assignment_2_report.pdf     # Detailed analysis and methodology report
```

---

# 🛠️ Setup & Installation

This project uses Conda for dependency management to ensure reproducibility.

## 1. Clone the Repository

```bash
git clone <repository-url>
cd repo
```

## 2. Create the Conda Environment

```bash
conda env create -f environment.yml
```

## 3. Activate the Environment

```bash
conda activate gans
```

## 4. Navigate to the Model Directory

```bash
cd model
```

---

# 🚀 Training

Run the training script:

```bash
python train.py
```

---

# 🔮 Inference

## MLP

```bash
python predict.py -i ../data/example_input.txt -o ../data/output_mlp.txt -m mlp
```

## GAN

```bash
python predict.py -i ../data/example_input.txt -o ../data/output_gan.txt -m gan
```

## VAE

```bash
python predict.py -i ../data/example_input.txt -o ../data/output_vae.txt -m vae
```

## Transformer

```bash
python predict.py -i ../data/example_input.txt -o ../data/output_transformer.txt -m transformer
```
