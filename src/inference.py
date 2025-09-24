"""
Inference script for Urdu to Roman transliteration.
"""
import torch
import argparse
from .transliterator import Transliterator


def main():
    parser = argparse.ArgumentParser(description='Urdu to Roman Transliteration Inference')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to tokenizer')
    parser.add_argument('--device', type=str, default='auto', help='Device (cpu/cuda/auto)')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    parser.add_argument('--text', type=str, help='Urdu text to transliterate')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Loading model from {args.model_path}")
    print(f"Using device: {device}")
    
    # Load transliterator
    transliterator = Transliterator.load_from_checkpoint(
        args.model_path, args.tokenizer_path, device
    )
    
    if args.interactive:
        print("\nUrdu to Roman Transliteration")
        print("Type 'exit' to quit")
        print("-" * 40)
        
        while True:
            try:
                urdu_text = input("\nEnter Urdu text: ").strip()
                if urdu_text.lower() == 'exit':
                    break
                
                if urdu_text:
                    transliteration = transliterator.transliterate(urdu_text)
                    print(f"Transliteration: {transliteration}")
                    
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    elif args.text:
        transliteration = transliterator.transliterate(args.text)
        print(f"Input: {args.text}")
        print(f"Output: {transliteration}")
    
    else:
        print("Please provide either --text or use --interactive mode")


if __name__ == "__main__":
    main()
