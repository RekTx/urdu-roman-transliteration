"""
BPE Tokenizer implementation from scratch for Urdu-Roman transliteration.
"""
import re
import json
from collections import defaultdict, Counter


class BPETokenizer:
    """Byte Pair Encoding tokenizer implementation from scratch."""
    
    def __init__(self, vocab_size=10000, special_tokens=None):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
        self.vocab = {}
        self.merges = []
        self.word_freqs = {}
        self.token_to_id = {}
        self.id_to_token = {}
    
    def train(self, texts):
        """Train the BPE tokenizer on given texts."""
        # Initialize vocabulary with characters
        vocab = set()
        for text in texts:
            for word in text.split():
                word = " ".join(list(word)) + " </w>"
                vocab.update(word.split())
        
        # Add special tokens
        for token in self.special_tokens:
            vocab.add(token)
        
        # Create initial vocabulary
        self.vocab = list(vocab)
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for i, token in enumerate(self.vocab)}
        
        # Count word frequencies
        word_freqs = Counter()
        for text in texts:
            for word in text.split():
                word = " ".join(list(word)) + " </w>"
                word_freqs[word] += 1
        
        # Perform BPE merges
        num_merges = self.vocab_size - len(self.vocab)
        for i in range(num_merges):
            # Find most frequent pair
            pairs = self.get_stats(word_freqs)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            
            # Merge the most frequent pair
            word_freqs = self.merge_vocab(best_pair, word_freqs)
            self.merges.append(best_pair)
            
            # Update vocabulary
            new_token = best_pair[0] + best_pair[1]
            if new_token not in self.token_to_id:
                self.token_to_id[new_token] = len(self.token_to_id)
                self.id_to_token[len(self.id_to_token)] = new_token
    
    def get_stats(self, vocab):
        """Get statistics of byte pair occurrences."""
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols)-1):
                pairs[symbols[i], symbols[i+1]] += freq
        return pairs
    
    def merge_vocab(self, pair, vocab):
        """Merge vocabulary based on the given pair."""
        new_vocab = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?=!\S)')
        for word in vocab:
            new_word = p.sub(''.join(pair), word)
            new_vocab[new_word] = vocab[word]
        return new_vocab
    
    def encode(self, text):
        """Convert text to token IDs."""
        tokens = []
        for word in text.split():
            word = " ".join(list(word)) + " </w>"
            
            # Apply all merges
            for merge in self.merges:
                bigram = re.escape(' '.join(merge))
                p = re.compile(r'(?<!\S)' + bigram + r'(?=!\S)')
                word = p.sub(''.join(merge), word)
            
            # Split into tokens and convert to IDs
            subwords = word.split()
            for subword in subwords:
                if subword in self.token_to_id:
                    tokens.append(self.token_to_id[subword])
                else:
                    tokens.append(self.token_to_id["[UNK]"])
        
        return tokens
    
    def decode(self, ids):
        """Convert token IDs back to text."""
        tokens = []
        for id in ids:
            if id in self.id_to_token:
                token = self.id_to_token[id]
                # Remove </w> and spaces between subwords
                if token == "</w>":
                    continue
                if token.startswith("##"):
                    tokens.append(token[2:])
                else:
                    tokens.append(token.replace(" ", ""))
        
        # Combine tokens into words
        text = "".join(tokens)
        return text
    
    def get_special_token_ids(self):
        """Get special token IDs."""
        return {
            'PAD_ID': self.token_to_id.get("[PAD]", 0),
            'UNK_ID': self.token_to_id.get("[UNK]", 1),
            'CLS_ID': self.token_to_id.get("[CLS]", 2),
            'SEP_ID': self.token_to_id.get("[SEP]", 3)
        }
    
    def save(self, path):
        """Save the tokenizer to a file."""
        data = {
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
            "merges": self.merges,
            "token_to_id": self.token_to_id,
            "id_to_token": self.id_to_token
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
    
    @classmethod
    def load(cls, path):
        """Load a tokenizer from a file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tokenizer = cls(data["vocab_size"], data["special_tokens"])
        tokenizer.merges = [tuple(merge) for merge in data["merges"]]
        tokenizer.token_to_id = {k: v for k, v in data["token_to_id"].items()}
        tokenizer.id_to_token = {int(k): v for k, v in data["id_to_token"].items()}
        return tokenizer
