"""
Training utilities for the Urdu to Roman transliteration model.
"""
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time
import numpy as np
from typing import List, Dict, Tuple, Optional

from .model import Encoder, Decoder, init_decoder_hidden
from .evaluation import calculate_perplexity


class Seq2SeqDataset(Dataset):
    """Dataset for sequence-to-sequence learning."""
    
    def __init__(self, src_tensors: List[torch.Tensor], tgt_tensors: List[torch.Tensor]):
        self.src = src_tensors
        self.tgt = tgt_tensors

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx]


def create_data_loaders(source_texts: List[str], target_texts: List[str], 
                       tokenizer, batch_size: int = 32, 
                       train_ratio: float = 0.5, val_ratio: float = 0.25):
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        source_texts: List of source texts
        target_texts: List of target texts
        tokenizer: BPE tokenizer
        batch_size: Batch size
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Get special token IDs
    special_tokens = tokenizer.get_special_token_ids()
    PAD_ID = special_tokens['PAD_ID']
    CLS_ID = special_tokens['CLS_ID']
    SEP_ID = special_tokens['SEP_ID']
    
    # Encode sequences
    source_ids = [tokenizer.encode(text) for text in source_texts]
    target_ids = [tokenizer.encode(text) for text in target_texts]
    
    # Add special tokens
    source_ids = [[CLS_ID] + seq + [SEP_ID] for seq in source_ids]
    target_ids = [[CLS_ID] + seq + [SEP_ID] for seq in target_ids]
    
    # Convert to tensors
    source_tensors = [torch.tensor(seq, dtype=torch.long) for seq in source_ids]
    target_tensors = [torch.tensor(seq, dtype=torch.long) for seq in target_ids]
    
    # Pad sequences
    source_padded = rnn_utils.pad_sequence(source_tensors, batch_first=True, padding_value=PAD_ID)
    target_padded = rnn_utils.pad_sequence(target_tensors, batch_first=True, padding_value=PAD_ID)
    
    # Split dataset
    N = len(source_padded)
    train_end = int(N * train_ratio)
    val_end = train_end + int(N * val_ratio)
    
    # Create datasets
    train_dataset = Seq2SeqDataset(source_padded[:train_end], target_padded[:train_end])
    val_dataset = Seq2SeqDataset(source_padded[train_end:val_end], target_padded[train_end:val_end])
    test_dataset = Seq2SeqDataset(source_padded[val_end:], target_padded[val_end:])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader


def evaluate_model(encoder, decoder, dataloader, criterion, device, num_layers_dec: int):
    """
    Evaluate the model on a given dataset.
    
    Args:
        encoder: Encoder model
        decoder: Decoder model
        dataloader: Data loader
        criterion: Loss function
        device: PyTorch device
        num_layers_dec: Number of decoder layers
        
    Returns:
        Average loss
    """
    encoder.eval()
    decoder.eval()
    total_loss = 0
    
    # Get special token IDs (assuming they're available globally or passed)
    PAD_ID = 0  # This should be passed as parameter in real implementation

    with torch.no_grad():
        for batch_src, batch_tgt in dataloader:
            batch_src, batch_tgt = batch_src.to(device), batch_tgt.to(device)
            batch_src_len = (batch_src != PAD_ID).sum(dim=1)

            # Encoder forward pass
            enc_out, (enc_h, enc_c) = encoder(batch_src, batch_src_len)

            # Initialize decoder hidden state
            dec_h, dec_c = init_decoder_hidden(enc_h, enc_c, num_layers_dec)

            # First input to decoder is CLS token
            input_tok = batch_tgt[:, 0].unsqueeze(1)
            vocab_size = decoder.fc_out.out_features
            outputs = torch.zeros(batch_tgt.size(0), batch_tgt.size(1), vocab_size).to(device)

            # Decoder forward pass without teacher forcing
            for t in range(1, batch_tgt.size(1)):
                dec_out, (dec_h, dec_c) = decoder(input_tok, (dec_h, dec_c))
                outputs[:, t, :] = dec_out.squeeze(1)
                input_tok = dec_out.argmax(2)  # Use model's prediction as next input

            # Calculate loss
            outputs_flat = outputs[:, 1:].reshape(-1, vocab_size)
            target_flat = batch_tgt[:, 1:].reshape(-1)
            loss = criterion(outputs_flat, target_flat)

            total_loss += loss.item()

    return total_loss / len(dataloader)


def train_model(encoder, decoder, train_loader, val_loader, device, config: Dict):
    """
    Train the seq2seq model.
    
    Args:
        encoder: Encoder model
        decoder: Decoder model
        train_loader: Training data loader
        val_loader: Validation data loader
        device: PyTorch device
        config: Training configuration
        
    Returns:
        Dictionary with training history
    """
    # Get special token IDs
    PAD_ID = 0  # This should be passed as parameter
    CLS_ID = 2  # This should be passed as parameter
    SEP_ID = 3  # This should be passed as parameter
    
    # Training parameters
    num_epochs = config.get('num_epochs', 50)
    learning_rate = config.get('lr', 1e-3)
    teacher_forcing_ratio = config.get('teacher_forcing_ratio', 0.5)
    clip = config.get('gradient_clip', 1.0)
    patience = config.get('patience', 5)
    num_layers_dec = config.get('num_layers_dec', 4)
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), 
        lr=learning_rate
    )
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    
    # Early stopping
    best_val_loss = float('inf')
    counter = 0
    
    # Training history
    train_losses = []
    val_losses = []
    
    vocab_size = decoder.fc_out.out_features
    
    print(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        encoder.train()
        decoder.train()
        total_loss = 0
        epoch_start_time = time.time()
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_src, batch_tgt in progress_bar:
            batch_src, batch_tgt = batch_src.to(device), batch_tgt.to(device)
            batch_src_len = (batch_src != PAD_ID).sum(dim=1)
            
            optimizer.zero_grad()
            
            # Encoder forward pass
            enc_out, (enc_h, enc_c) = encoder(batch_src, batch_src_len)
            
            # Decoder initialization
            dec_h, dec_c = init_decoder_hidden(enc_h, enc_c, num_layers_dec)
            
            # First input to decoder is CLS token
            input_tok = batch_tgt[:, 0].unsqueeze(1)
            outputs = torch.zeros(batch_tgt.size(0), batch_tgt.size(1), vocab_size).to(device)
            
            # Decoder forward pass with teacher forcing
            for t in range(1, batch_tgt.size(1)):
                dec_out, (dec_h, dec_c) = decoder(input_tok, (dec_h, dec_c))
                outputs[:, t, :] = dec_out.squeeze(1)
                
                # Teacher forcing
                teacher_force = torch.rand(1).item() < teacher_forcing_ratio
                top1 = dec_out.argmax(2)
                input_tok = batch_tgt[:, t].unsqueeze(1) if teacher_force else top1
            
            # Loss calculation
            outputs_flat = outputs[:, 1:].reshape(-1, vocab_size)
            target_flat = batch_tgt[:, 1:].reshape(-1)
            loss = criterion(outputs_flat, target_flat)
            
            # Backpropagation
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(decoder.parameters()),
                clip
            )
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Calculate average loss for the epoch
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Validation
        val_loss = evaluate_model(encoder, decoder, val_loader, criterion, device, num_layers_dec)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {epoch_time:.2f}s")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'num_epochs_trained': len(train_losses)
    }


def save_checkpoint(encoder, decoder, optimizer, config, train_history, filepath: str):
    """
    Save model checkpoint.
    
    Args:
        encoder: Encoder model
        decoder: Decoder model
        optimizer: Optimizer
        config: Training configuration
        train_history: Training history
        filepath: Path to save checkpoint
    """
    torch.save({
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'optimizer': optimizer.state_dict(),
        'config': config,
        'train_history': train_history
    }, filepath)
    
    print(f"Model checkpoint saved to {filepath}")


def load_checkpoint(filepath: str, encoder, decoder, optimizer=None, device='cpu'):
    """
    Load model checkpoint.
    
    Args:
        filepath: Path to checkpoint
        encoder: Encoder model to load state into
        decoder: Decoder model to load state into
        optimizer: Optimizer to load state into (optional)
        device: PyTorch device
        
    Returns:
        Dictionary with config and training history
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
    
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    return {
        'config': checkpoint['config'],
        'train_history': checkpoint['train_history']
    }
