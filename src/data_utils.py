"""
Data preprocessing utilities for Urdu to Roman transliteration.
"""
import re
import os
import glob
from typing import List, Dict, Tuple


def normalize_urdu(text: str) -> str:
    """
    Normalize Urdu text:
    - Remove extra spaces
    - Normalize certain characters
    - Remove unwanted diacritics
    """
    # Remove unwanted punctuation and diacritics
    text = re.sub(r"[ۂٰٖؔؕ]", "", text)

    # Replace Arabic-Indic digits with normal digits
    text = re.sub(r"[۰-۹]", lambda x: str(ord(x.group()) - 1776), text)

    # Normalize spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def clean_roman_safe(text: str) -> str:
    """
    Clean Roman Urdu text while keeping diacritics:
    - Lowercase
    - Normalize spaces
    - Replace fancy quotes/dashes
    """
    text = text.lower()
    text = text.replace("'", "'").replace("'", "'")
    text = text.replace(""", '"').replace(""", '"')
    text = text.replace("–", "-").replace("—", "-")
    text = text.replace("…", "...")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_rekhta_dataset(data_dir: str) -> List[Dict[str, str]]:
    """
    Load Urdu-Roman pairs from the Rekhta dataset.
    
    Args:
        data_dir: Path to the dataset directory
        
    Returns:
        List of dictionaries containing Urdu-Roman pairs
    """
    pairs = []
    
    # Get all author directories
    authors = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    for author in authors:
        ur_path = os.path.join(data_dir, author, "ur")
        en_path = os.path.join(data_dir, author, "en")

        if not os.path.exists(ur_path) or not os.path.exists(en_path):
            continue

        ur_files = glob.glob(os.path.join(ur_path, "*"))

        for ufile in ur_files:
            fname = os.path.basename(ufile)
            efile = os.path.join(en_path, fname)

            if not os.path.exists(efile):
                continue

            try:
                with open(ufile, "r", encoding="utf-8") as f:
                    ur_lines = f.read().splitlines()

                with open(efile, "r", encoding="utf-8") as f:
                    en_lines = f.read().splitlines()

                # Ensure same number of lines
                if len(ur_lines) != len(en_lines):
                    continue

                # Store pairs
                for ur, en in zip(ur_lines, en_lines):
                    if ur.strip() and en.strip():  # Skip empty lines
                        pairs.append({
                            "ur_raw": ur.strip(),
                            "en_raw": en.strip(),
                            "author": author,
                            "file": fname
                        })
            except Exception as e:
                print(f"Error processing {ufile}: {e}")
                continue

    return pairs


def preprocess_dataset(pairs: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Preprocess the dataset by normalizing Urdu and cleaning Roman text.
    
    Args:
        pairs: List of raw Urdu-Roman pairs
        
    Returns:
        List of preprocessed pairs
    """
    processed_pairs = []
    
    for pair in pairs:
        # Normalize Urdu text
        ur_norm = normalize_urdu(pair["ur_raw"])
        
        # Clean Roman text
        en_clean = clean_roman_safe(pair["en_raw"])
        
        # Skip very short or very long sequences
        if len(ur_norm.split()) < 1 or len(ur_norm.split()) > 50:
            continue
        if len(en_clean.split()) < 1 or len(en_clean.split()) > 50:
            continue
        
        # Create processed pair
        processed_pair = {
            "ur_raw": pair["ur_raw"],
            "en_raw": pair["en_raw"],
            "ur_norm": ur_norm,
            "en_clean": en_clean,
            "author": pair.get("author", "unknown"),
            "file": pair.get("file", "unknown")
        }
        
        processed_pairs.append(processed_pair)
    
    return processed_pairs


def split_dataset(pairs: List[Dict[str, str]], 
                 train_ratio: float = 0.5, 
                 val_ratio: float = 0.25) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        pairs: List of data pairs
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set (remaining goes to test)
        
    Returns:
        Tuple of (train_pairs, val_pairs, test_pairs)
    """
    n = len(pairs)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_pairs = pairs[:train_end]
    val_pairs = pairs[train_end:val_end]
    test_pairs = pairs[val_end:]
    
    return train_pairs, val_pairs, test_pairs


def save_processed_data(pairs: List[Dict[str, str]], output_dir: str):
    """
    Save processed data to text files.
    
    Args:
        pairs: List of processed pairs
        output_dir: Output directory path
    """
    os.makedirs(output_dir, exist_ok=True)
    
    source_path = os.path.join(output_dir, "source.txt")
    target_path = os.path.join(output_dir, "target.txt")
    
    with open(source_path, "w", encoding="utf-8") as fs, \
         open(target_path, "w", encoding="utf-8") as ft:
        for pair in pairs:
            fs.write(pair["ur_norm"] + "\n")
            ft.write(pair["en_clean"] + "\n")
    
    print(f"Saved {len(pairs)} pairs to {output_dir}")


def load_processed_data(data_dir: str) -> Tuple[List[str], List[str]]:
    """
    Load processed data from text files.
    
    Args:
        data_dir: Directory containing source.txt and target.txt
        
    Returns:
        Tuple of (source_texts, target_texts)
    """
    source_path = os.path.join(data_dir, "source.txt")
    target_path = os.path.join(data_dir, "target.txt")
    
    with open(source_path, "r", encoding="utf-8") as f:
        source_texts = [line.strip() for line in f.readlines()]
    
    with open(target_path, "r", encoding="utf-8") as f:
        target_texts = [line.strip() for line in f.readlines()]
    
    assert len(source_texts) == len(target_texts), "Source and target files must have same number of lines"
    
    return source_texts, target_texts
