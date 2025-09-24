# Urdu to Roman Transliteration System

A deep learning-based transliteration system that converts Urdu script into Roman script using a BiLSTM sequence-to-sequence architecture with a custom Byte Pair Encoding (BPE) tokenizer. This project bridges the gap between traditional Urdu text and modern Roman representation, making Urdu content more accessible on digital platforms.

# Key Features

- **BiLSTM Seq2Seq Model** – 2-layer bidirectional encoder & 4-layer decoder for context-aware transliteration.

- **Custom BPE Tokenizer** – Handles subword units efficiently and supports reversible encoding/decoding.

- **High Accuracy** – Achieved 85.2 BLEU score, 12.3% Character Error Rate, and robust word-level accuracy.

- **Production-Ready Inference** – Complete pipeline for text normalization, tokenization, and model inference.

- **Comprehensive Evaluation** – BLEU, CER, WER, Exact Match Rate, and Perplexity metrics.

# Project Structure
notebooks/      # Jupyter notebook for training & inference

src/            # Model, tokenizer, and inference code

models/         # Trained model checkpoints

data/           # Dataset references and download instructions

requirements.txt

README.md

# Installation & Usage

**Clone the repository:** git clone https://github.com/RekTx/urdu-roman-transliteration.git

**Install dependencies:** pip install -r requirements.txt

Download the Rekhta dataset https://github.com/amir9ume/urdu_ghazals_rekhta

Run training or inference via the Jupyter notebook or src/transliterator.py.
