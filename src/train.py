"""
Main training script for Urdu to Roman transliteration.
"""
import torch
import argparse
import os
import json
from tqdm import tqdm

from .data_utils import load_rekhta_dataset, preprocess_dataset, split_dataset
from .tokenizer import BPETokenizer
from .model import Encoder, Decoder
from .train_utils import create_data_loaders, train_model, save_checkpoint
from .transliterator import Transliterator
from .evaluation import evaluate_comprehensive, print_evaluation_summary


def main():
    parser = argparse.ArgumentParser(description='Train Urdu to Roman transliteration model')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='./models', help='Output directory for models')
    parser.add_argument('--embed_size', type=int, default=256, help='Embedding size')
    parser.add_argument('--hidden_size', type=int, default=512, help='Hidden size')
    parser.add_argument('--num_layers_enc', type=int, default=2, help='Number of encoder layers')
    parser.add_argument('--num_layers_dec', type=int, default=4, help='Number of decoder layers')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--vocab_size', type=int, default=10000, help='Vocabulary size')
    parser.add_argument('--device', type=str, default='auto', help='Device (cpu/cuda/auto)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and preprocess data
    print("Loading dataset...")
    pairs = load_rekhta_dataset(args.data_dir)
    print(f"Loaded {len(pairs)} pairs")
    
    print("Preprocessing dataset...")
    processed_pairs = preprocess_dataset(pairs)
    print(f"After preprocessing: {len(processed_pairs)} pairs")
    
    # Extract texts
    source_texts = [p["ur_norm"] for p in processed_pairs]
    target_texts = [p["en_clean"] for p in processed_pairs]
    all_texts = source_texts + target_texts
    
    # Train tokenizer
    print("Training BPE tokenizer...")
    tokenizer = BPETokenizer(vocab_size=args.vocab_size)
    tokenizer.train(all_texts)
    
    # Save tokenizer
    tokenizer_path = os.path.join(args.output_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    print(f"Tokenizer saved to {tokenizer_path}")
    
    # Get special token IDs and vocab size
    special_tokens = tokenizer.get_special_token_ids()
    vocab_size = len(tokenizer.token_to_id)
    print(f"Vocabulary size: {vocab_size}")
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        source_texts, target_texts, tokenizer, batch_size=args.batch_size
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Initialize models
    encoder = Encoder(
        vocab_size=vocab_size,
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers_enc,
        dropout=args.dropout,
        pad_idx=special_tokens['PAD_ID']
    )
    
    decoder = Decoder(
        vocab_size=vocab_size,
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers_dec,
        dropout=args.dropout,
        pad_idx=special_tokens['PAD_ID']
    )
    
    # Move to device
    encoder.to(device)
    decoder.to(device)
    
    # Training configuration
    config = {
        'embed_size': args.embed_size,
        'hidden_size': args.hidden_size,
        'num_layers_enc': args.num_layers_enc,
        'num_layers_dec': args.num_layers_dec,
        'dropout': args.dropout,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'vocab_size': vocab_size
    }
    
    # Train model
    print("Starting training...")
    train_history = train_model(encoder, decoder, train_loader, val_loader, device, config)
    
    # Save final model
    model_path = os.path.join(args.output_dir, "best_model.pt")
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), 
        lr=args.lr
    )
    save_checkpoint(encoder, decoder, optimizer, config, train_history, model_path)
    
    # Create transliterator and evaluate
    print("Creating transliterator...")
    transliterator = Transliterator(encoder, decoder, tokenizer, device, args.num_layers_dec)
    
    # Split data for evaluation
    train_pairs, val_pairs, test_pairs = split_dataset(processed_pairs)
    
    print("Evaluating on test set...")
    results = evaluate_comprehensive(transliterator, test_pairs, num_samples=500)
    
    # Print results
    print_evaluation_summary(results, "Final Model")
    
    # Save evaluation results
    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        # Convert numpy types to Python types for JSON serialization
        json_results = {
            'metrics': {
                'bleu': float(results['avg_bleu']),
                'cer': float(results['avg_cer']),
                'wer': float(results['avg_wer']),
                'exact_match_rate': float(results['exact_match_rate'])
            },
            'config': config,
            'train_history': train_history
        }
        json.dump(json_results, f, indent=2)
    
    print(f"Evaluation results saved to {results_path}")
    print("Training completed!")


if __name__ == "__main__":
    main()
