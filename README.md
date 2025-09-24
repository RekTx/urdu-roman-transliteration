# Urdu to Roman Transliteration

This project implements a BiLSTM Seq2Seq model for transliterating Urdu text to Roman script using PyTorch.

## Features

- **Custom BPE Tokenizer**: Built from scratch for Urdu-Roman text processing
- **BiLSTM Encoder**: 2-layer bidirectional LSTM encoder
- **Multi-layer Decoder**: 4-layer LSTM decoder with teacher forcing
- **Comprehensive Evaluation**: BLEU, CER, WER, and perplexity metrics
- **Modular Design**: Separated into reusable modules

## Installation

1. Clone the repository and navigate to the project directory:
```bash
cd "e:\NLP\Ass 1"
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

## Project Structure

```
src/
├── __init__.py           # Package initialization
├── tokenizer.py          # BPE tokenizer implementation
├── model.py             # BiLSTM Seq2Seq model architecture
├── transliterator.py    # Transliteration inference class
├── data_utils.py        # Data preprocessing utilities
├── evaluation.py        # Evaluation metrics
├── train_utils.py       # Training utilities
├── train.py            # Main training script
└── inference.py        # Inference script
```

## Usage

### Training a New Model

```bash
python -m src.train --data_dir "data/raw/rekhta/dataset/dataset" --output_dir "./models"
```

Training options:
- `--embed_size`: Embedding dimension (default: 256)
- `--hidden_size`: Hidden layer size (default: 512)
- `--num_layers_enc`: Encoder layers (default: 2)
- `--num_layers_dec`: Decoder layers (default: 4)
- `--dropout`: Dropout rate (default: 0.3)
- `--lr`: Learning rate (default: 5e-4)
- `--batch_size`: Batch size (default: 64)
- `--num_epochs`: Training epochs (default: 50)

### Running Inference

#### Interactive Mode
```bash
python -m src.inference --model_path "models/best_model.pt" --tokenizer_path "models/tokenizer.json" --interactive
```

#### Single Text Translation
```bash
python -m src.inference --model_path "models/best_model.pt" --tokenizer_path "models/tokenizer.json" --text "میرا نام احمد ہے"
```

### Using as a Python Module

```python
from src.transliterator import Transliterator

# Load trained model
transliterator = Transliterator.load_from_checkpoint(
    "models/best_model.pt", 
    "models/tokenizer.json"
)

# Transliterate text
urdu_text = "میرا نام احمد ہے"
roman_text = transliterator.transliterate(urdu_text)
print(f"Urdu: {urdu_text}")
print(f"Roman: {roman_text}")
```

### Training with Custom Data

```python
from src.data_utils import load_rekhta_dataset, preprocess_dataset
from src.tokenizer import BPETokenizer
from src.model import Encoder, Decoder
from src.train_utils import create_data_loaders, train_model

# Load and preprocess data
pairs = load_rekhta_dataset("path/to/dataset")
processed_pairs = preprocess_dataset(pairs)

# Extract texts
source_texts = [p["ur_norm"] for p in processed_pairs]
target_texts = [p["en_clean"] for p in processed_pairs]

# Train tokenizer
tokenizer = BPETokenizer(vocab_size=10000)
tokenizer.train(source_texts + target_texts)

# Create data loaders
train_loader, val_loader, test_loader = create_data_loaders(
    source_texts, target_texts, tokenizer, batch_size=32
)

# Initialize and train models
# ... (see train.py for complete example)
```

## Model Architecture

- **Encoder**: 2-layer bidirectional LSTM with projection layers
- **Decoder**: 4-layer unidirectional LSTM with attention mechanism
- **Embedding**: Learned embeddings for source and target vocabularies
- **Output**: Linear layer projecting to vocabulary size

## Evaluation Metrics

- **BLEU Score**: Measures n-gram overlap with reference
- **Character Error Rate (CER)**: Character-level edit distance
- **Word Error Rate (WER)**: Word-level edit distance  
- **Exact Match Rate**: Percentage of perfect matches
- **Perplexity**: Model confidence measure

## Dataset

The project uses the Rekhta dataset containing Urdu poetry with Roman transliterations from various poets.

## Requirements

- Python 3.7+
- PyTorch 1.9+
- NumPy
- tqdm
- matplotlib (for visualization)

## License

This project is for educational purposes. Please respect the original dataset licenses.
