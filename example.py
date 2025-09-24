"""
Example usage of the Urdu to Roman transliteration model.
"""
import torch
import os
from src.data_utils import load_rekhta_dataset, preprocess_dataset, split_dataset
from src.tokenizer import BPETokenizer
from src.model import Encoder, Decoder
from src.transliterator import Transliterator
from src.evaluation import evaluate_comprehensive, print_evaluation_summary, show_examples


def quick_example():
    """Quick example showing basic usage."""
    
    # Check if models exist
    model_path = "urdu_to_roman_bilstm_model.pt"
    tokenizer_path = "data/processed/bpe_tokenizer.json"
    
    if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
        print("Model files not found. Please train a model first using the notebook or train.py")
        return
    
    print("Loading trained model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Load the model from the notebook's saved file
        checkpoint = torch.load(model_path, map_location=device)
        tokenizer = BPETokenizer.load(tokenizer_path)
        
        # Get configuration
        config = checkpoint['config']
        vocab_size = checkpoint['vocab_size']
        
        # Initialize models
        encoder = Encoder(
            vocab_size=vocab_size,
            embed_size=config['embed_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers_enc'],
            dropout=config['dropout'],
            pad_idx=checkpoint['PAD_ID']
        )
        
        decoder = Decoder(
            vocab_size=vocab_size,
            embed_size=config['embed_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers_dec'],
            dropout=config['dropout'],
            pad_idx=checkpoint['PAD_ID']
        )
        
        # Load weights
        encoder.load_state_dict(checkpoint['encoder'])
        decoder.load_state_dict(checkpoint['decoder'])
        
        # Move to device
        encoder.to(device)
        decoder.to(device)
        
        # Create transliterator
        transliterator = Transliterator(
            encoder=encoder,
            decoder=decoder,
            tokenizer=tokenizer,
            device=device,
            num_layers_dec=config['num_layers_dec']
        )
        
        # Test examples
        test_examples = [
            "میرا نام احمد ہے",
            "آج موسم بہت اچھا ہے",
            "میں اردو بولتا ہوں",
            "یہ کتاب بہت دلچسپ ہے",
            "وہ اسکول جا رہا ہے"
        ]
        
        print("\nTransliteration Examples:")
        print("=" * 50)
        
        for example in test_examples:
            transliteration = transliterator.transliterate(example)
            print(f"Urdu:  {example}")
            print(f"Roman: {transliteration}")
            print("-" * 30)
        
        print("\nModel loaded and working successfully!")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please make sure you have run the notebook to train and save the model.")


def evaluate_example():
    """Example showing how to evaluate the model."""
    
    data_dir = "data/raw/rekhta/dataset/dataset"
    if not os.path.exists(data_dir):
        print(f"Dataset directory not found: {data_dir}")
        return
    
    print("Loading and preprocessing dataset for evaluation...")
    
    # Load data
    pairs = load_rekhta_dataset(data_dir)
    processed_pairs = preprocess_dataset(pairs)
    
    # Split data
    train_pairs, val_pairs, test_pairs = split_dataset(processed_pairs)
    
    print(f"Test pairs available: {len(test_pairs)}")
    
    # Load model if available
    model_path = "urdu_to_roman_bilstm_model.pt"
    tokenizer_path = "data/processed/bpe_tokenizer.json"
    
    if os.path.exists(model_path) and os.path.exists(tokenizer_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            # Load using the transliterator's class method would be ideal
            # For now, let's just show the data structure
            print("\nSample test pairs:")
            for i, pair in enumerate(test_pairs[:5]):
                print(f"\nPair {i+1}:")
                print(f"Urdu:  {pair['ur_norm']}")
                print(f"Roman: {pair['en_clean']}")
            
        except Exception as e:
            print(f"Could not load model for evaluation: {e}")
    else:
        print("Model files not found for evaluation.")


def train_example():
    """Example showing how to train a model from scratch."""
    
    print("This example shows how to train a model from scratch.")
    print("For actual training, please use:")
    print("python -m src.train --data_dir 'data/raw/rekhta/dataset/dataset' --output_dir './models'")
    
    # Show configuration example
    config_example = {
        'embed_size': 256,
        'hidden_size': 512,
        'num_layers_enc': 2,
        'num_layers_dec': 4,
        'dropout': 0.3,
        'lr': 5e-4,
        'batch_size': 64,
        'num_epochs': 50,
        'vocab_size': 10000
    }
    
    print("\nExample training configuration:")
    for key, value in config_example.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    print("Urdu to Roman Transliteration - Examples")
    print("=" * 50)
    
    while True:
        print("\nChoose an example:")
        print("1. Quick transliteration example")
        print("2. Evaluation example")
        print("3. Training configuration example")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            quick_example()
        elif choice == '2':
            evaluate_example()
        elif choice == '3':
            train_example()
        elif choice == '4':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")
