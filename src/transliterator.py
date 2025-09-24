"""
Transliterator class for converting Urdu text to Roman script using trained models.
"""
import re
import torch
from .tokenizer import BPETokenizer
from .model import Encoder, Decoder


class Transliterator:
    """Urdu to Roman transliterator using BiLSTM Seq2Seq model."""
    
    def __init__(self, encoder, decoder, tokenizer, device, num_layers_dec=4):
        """
        Initialize the transliterator.
        
        Args:
            encoder: Trained encoder model
            decoder: Trained decoder model
            tokenizer: BPE tokenizer
            device: PyTorch device
            num_layers_dec: Number of decoder layers
        """
        self.encoder = encoder
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.device = device
        self.num_layers_dec = num_layers_dec
        
        # Get special token IDs
        special_tokens = self.tokenizer.get_special_token_ids()
        self.PAD_ID = special_tokens['PAD_ID']
        self.UNK_ID = special_tokens['UNK_ID']
        self.CLS_ID = special_tokens['CLS_ID']
        self.SEP_ID = special_tokens['SEP_ID']

    def normalize_urdu(self, text):
        """
        Normalize Urdu text.
        
        Args:
            text: Input Urdu text
            
        Returns:
            Normalized Urdu text
        """
        # Remove extra diacritics
        text = re.sub(r"[ۂٰٖؔؕ]", "", text)
        
        # Replace Arabic-Indic digits with normal digits
        text = re.sub(r"[۰-۹]", lambda x: str(ord(x.group()) - 1776), text)
        
        # Normalize spaces
        text = re.sub(r"\s+", " ", text).strip()
        
        return text

    def init_decoder_hidden(self, enc_h, enc_c):
        """
        Initialize decoder hidden state from encoder hidden state.
        
        Args:
            enc_h: Encoder hidden state
            enc_c: Encoder cell state
            
        Returns:
            Decoder initial hidden and cell states
        """
        # enc_h and enc_c should be [batch, hidden_size] after encoder projection
        # Repeat for all decoder layers (identity mapping)
        dec_h = enc_h.unsqueeze(0).repeat(self.num_layers_dec, 1, 1)
        dec_c = enc_c.unsqueeze(0).repeat(self.num_layers_dec, 1, 1)
        
        return dec_h, dec_c

    def transliterate(self, urdu_text, max_len=50):
        """
        Transliterate Urdu text to Roman script.
        
        Args:
            urdu_text: Input Urdu text
            max_len: Maximum output length
            
        Returns:
            Roman transliteration
        """
        self.encoder.eval()
        self.decoder.eval()

        # Normalize and tokenize
        urdu_norm = self.normalize_urdu(urdu_text)
        tokens = [self.CLS_ID] + self.tokenizer.encode(urdu_norm) + [self.SEP_ID]
        src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(self.device)
        src_len = torch.LongTensor([len(tokens)]).to(self.device)

        with torch.no_grad():
            # Encoder forward pass
            enc_out, (enc_h, enc_c) = self.encoder(src_tensor, src_len)

            # Initialize decoder hidden state
            dec_h, dec_c = self.init_decoder_hidden(enc_h, enc_c)

            # Start with CLS token
            input_tok = torch.LongTensor([[self.CLS_ID]]).to(self.device)

            result = []

            # Generate transliteration
            for t in range(max_len):
                dec_out, (dec_h, dec_c) = self.decoder(input_tok, (dec_h, dec_c))
                pred_token = dec_out.argmax(2).item()

                if pred_token == self.SEP_ID:
                    break

                result.append(pred_token)
                input_tok = torch.LongTensor([[pred_token]]).to(self.device)

        # Convert token IDs to text
        roman_text = self.tokenizer.decode(result)
        return roman_text
    
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, tokenizer_path, device='cpu'):
        """
        Load a transliterator from saved checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint
            tokenizer_path: Path to tokenizer file
            device: PyTorch device
            
        Returns:
            Loaded transliterator instance
        """
        # Load tokenizer
        tokenizer = BPETokenizer.load(tokenizer_path)
        special_tokens = tokenizer.get_special_token_ids()
        vocab_size = len(tokenizer.token_to_id)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = checkpoint['config']
        
        # Initialize models
        encoder = Encoder(
            vocab_size=vocab_size,
            embed_size=config['embed_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers_enc'],
            dropout=config['dropout'],
            pad_idx=special_tokens['PAD_ID']
        )
        
        decoder = Decoder(
            vocab_size=vocab_size,
            embed_size=config['embed_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers_dec'],
            dropout=config['dropout'],
            pad_idx=special_tokens['PAD_ID']
        )
        
        # Load state dictionaries
        encoder.load_state_dict(checkpoint['encoder'])
        decoder.load_state_dict(checkpoint['decoder'])
        
        # Move to device
        encoder.to(device)
        decoder.to(device)
        
        # Create transliterator
        transliterator = cls(
            encoder=encoder,
            decoder=decoder,
            tokenizer=tokenizer,
            device=device,
            num_layers_dec=config['num_layers_dec']
        )
        
        return transliterator
