"""
BiLSTM Seq2Seq model for Urdu to Roman transliteration.
"""
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils


class Encoder(nn.Module):
    """Bidirectional LSTM Encoder."""
    
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=2, dropout=0.1, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        # Bidirectional LSTM as required
        self.lstm = nn.LSTM(
            embed_size, hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout, bidirectional=True
        )
        # Projection layers for combining bidirectional states
        self.h_projection = nn.Linear(2 * hidden_size, hidden_size)
        self.c_projection = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, src, src_len):
        """
        Forward pass through the encoder.
        
        Args:
            src: [batch, src_len] - Source sequences
            src_len: Actual lengths for packing
            
        Returns:
            out: Encoded outputs
            (h_projected, c_projected): Final hidden states
        """
        emb = self.embedding(src)
        packed = rnn_utils.pack_padded_sequence(emb, src_len.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (h, c) = self.lstm(packed)
        out, _ = rnn_utils.pad_packed_sequence(packed_out, batch_first=True)
        
        # Combine forward and backward hidden states
        # h and c are [2*num_layers, batch, hidden_size]
        # We take the last layer's forward and backward states
        h_forward = h[-2, :, :]  # Last layer forward
        h_backward = h[-1, :, :]  # Last layer backward
        h_combined = torch.cat((h_forward, h_backward), dim=1)  # [batch, 2*hidden_size]
        h_projected = self.h_projection(h_combined)  # [batch, hidden_size]
        
        c_forward = c[-2, :, :]
        c_backward = c[-1, :, :]
        c_combined = torch.cat((c_forward, c_backward), dim=1)
        c_projected = self.c_projection(c_combined)
        
        return out, (h_projected, c_projected)


class Decoder(nn.Module):
    """Multi-layer LSTM Decoder."""
    
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=4, dropout=0.1, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        # 4 stacked LSTM layers as required
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout)
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, tgt, hidden):
        """
        Forward pass through the decoder.
        
        Args:
            tgt: [batch, tgt_len] - Target sequences
            hidden: Tuple (h, c) each [num_layers, batch, hidden_size]
            
        Returns:
            logits: Output logits
            hidden: Updated hidden states
        """
        emb = self.embedding(tgt)
        out, hidden = self.lstm(emb, hidden)
        logits = self.fc_out(out)
        return logits, hidden


class Seq2SeqModel(nn.Module):
    """Complete Seq2Seq model combining encoder and decoder."""
    
    def __init__(self, vocab_size, embed_size, hidden_size, 
                 num_layers_enc=2, num_layers_dec=4, dropout=0.1, pad_idx=0):
        super().__init__()
        self.encoder = Encoder(vocab_size, embed_size, hidden_size, num_layers_enc, dropout, pad_idx)
        self.decoder = Decoder(vocab_size, embed_size, hidden_size, num_layers_dec, dropout, pad_idx)
        self.num_layers_dec = num_layers_dec
    
    def init_decoder_hidden(self, enc_h, enc_c):
        """
        Initialize decoder hidden state from encoder final state.
        Using identity mapping as required.
        
        Args:
            enc_h: Encoder final hidden state [batch, hidden_size]
            enc_c: Encoder final cell state [batch, hidden_size]
            
        Returns:
            dec_h, dec_c: Decoder initial hidden states
        """
        # enc_h and enc_c are now [batch, hidden_size] (after projection)
        # Repeat for all decoder layers
        dec_h = enc_h.unsqueeze(0).repeat(self.num_layers_dec, 1, 1)
        dec_c = enc_c.unsqueeze(0).repeat(self.num_layers_dec, 1, 1)
        return dec_h, dec_c
    
    def forward(self, src, src_len, tgt=None, teacher_forcing_ratio=0.5):
        """
        Forward pass through the complete model.
        
        Args:
            src: Source sequences
            src_len: Source sequence lengths
            tgt: Target sequences (for training)
            teacher_forcing_ratio: Probability of using teacher forcing
            
        Returns:
            outputs: Model predictions
        """
        batch_size = src.size(0)
        max_len = tgt.size(1) if tgt is not None else 50
        vocab_size = self.decoder.fc_out.out_features
        
        # Encoder forward pass
        enc_out, (enc_h, enc_c) = self.encoder(src, src_len)
        
        # Initialize decoder hidden state
        dec_h, dec_c = self.init_decoder_hidden(enc_h, enc_c)
        
        # Initialize outputs tensor
        outputs = torch.zeros(batch_size, max_len, vocab_size).to(src.device)
        
        # First input to decoder (typically CLS token)
        if tgt is not None:
            input_tok = tgt[:, 0].unsqueeze(1)
        else:
            # For inference, start with CLS token (assuming it's index 2)
            input_tok = torch.full((batch_size, 1), 2, dtype=torch.long).to(src.device)
        
        # Decoder forward pass
        for t in range(1, max_len):
            dec_out, (dec_h, dec_c) = self.decoder(input_tok, (dec_h, dec_c))
            outputs[:, t, :] = dec_out.squeeze(1)
            
            # Determine next input
            if tgt is not None and torch.rand(1).item() < teacher_forcing_ratio:
                # Teacher forcing: use ground truth
                input_tok = tgt[:, t].unsqueeze(1)
            else:
                # Use model's prediction
                input_tok = dec_out.argmax(2)
        
        return outputs


def init_decoder_hidden(enc_h, enc_c, num_layers_dec):
    """
    Standalone function to initialize decoder hidden state from encoder final state.
    Using identity mapping as required.
    """
    # enc_h and enc_c are now [batch, hidden_size] (after projection)
    # Repeat for all decoder layers
    dec_h = enc_h.unsqueeze(0).repeat(num_layers_dec, 1, 1)
    dec_c = enc_c.unsqueeze(0).repeat(num_layers_dec, 1, 1)
    return dec_h, dec_c
