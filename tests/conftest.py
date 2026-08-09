import os
import json
import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock

class StubTokenizer:
    def __call__(self, text, return_tensors=None, padding=False, truncation=False, **kwargs):
        if isinstance(text, str):
            text = [text]
            
        batch_size = len(text)
        seq_length = 8  # Arbitrary dummy sequence length for the ONNX trace
        
        # Return the exact dictionary format SentenceTransformer (and ONNX) expects
        return {
            "input_ids": torch.ones(batch_size, seq_length, dtype=torch.long),
            "attention_mask": torch.ones(batch_size, seq_length, dtype=torch.long),
            "token_type_ids": torch.zeros(batch_size, seq_length, dtype=torch.long)
        }

    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        
        # A minimal, valid schema required by the Rust `tokenizers` library
        minimal_tokenizer_json = {
            "version": "1.0",
            "model": {
                "type": "WordPiece",
                "unk_token": "[UNK]",
                "continuing_subword_prefix": "##",
                "max_input_chars_per_word": 100,
                "vocab": {
                    "[UNK]": 0,
                    "[CLS]": 1,
                    "[SEP]": 2,
                    "[PAD]": 3,
                    "dummy": 4
                }
            }
        }
        
        with open(os.path.join(save_directory, "tokenizer.json"), "w") as f:
            json.dump(minimal_tokenizer_json, f)

class StubSentenceTransformer(nn.Module):
    def __init__(self, hidden_dim=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.similarity_fn_name = None
        self.tokenizer = StubTokenizer()
        
        # Mock the metadata properties expected by LanceBundleBuilder
        self.model_card_data = MagicMock()
        self.model_card_data.to_dict.return_value = {"base_model": "test-stub-model"}
        
        # A trivial linear layer just so ONNX has some math to trace
        self.dummy_projection = nn.Linear(1, self.hidden_dim)

    def get_max_seq_length(self):
        return 128

    def forward(self, features):
        # Grab the input tokens (shape: [batch_size, seq_length])
        x = features["input_ids"].float().unsqueeze(-1) # [batch, seq, 1]
        
        # Do trivial math to project to our expected vector dimension
        x = self.dummy_projection(x) # [batch, seq, hidden_dim]
        
        # Sum across the sequence length to simulate pooling
        sentence_embedding = x.sum(dim=1) # [batch, hidden_dim]
        
        return {"sentence_embedding": sentence_embedding}

@pytest.fixture(scope="session")
def dummy_model():
    """Yields the stub model instantly without network calls."""
    return StubSentenceTransformer(hidden_dim=4)

@pytest.fixture(scope="session")
def dummy_data():
    """Generates 4-dimensional dummy data to match the stub model."""
    texts = [
        "Quantum mechanics is the foundation of modern physics.",
        "Photosynthesis allows plants to convert sunlight into food.",
        "I love eating pepperoni pizza on Fridays."
    ]
    # Fake 4D embedding vectors corresponding to the texts
    vectors = [
        [0.1, 0.1, 0.1, 0.1], 
        [0.5, 0.5, 0.5, 0.5], 
        [0.9, 0.9, 0.9, 0.9]
    ]
    return texts, vectors
