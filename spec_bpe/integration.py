import os
import json
from typing import List, Optional, Union
from .tokenizer import SpecTokenizer

class SpecTokenizerTransformerWrapper:
    """
    A wrapper to make SpecTokenizer compatible with common interfaces.
    """
    def __init__(self, tokenizer: SpecTokenizer):
        self.tokenizer = tokenizer
        self.vocab = {k: v.decode('utf-8', errors='replace') for k, v in tokenizer.vocab.items()}
        self.decoder = {v: k for k, v in self.vocab.items()}

    def tokenize(self, text: str) -> List[str]:
        ids = self.tokenizer.encode(text)
        return [self.vocab.get(idx, "[UNK]") for idx in ids]

    def encode(self, text: str, add_special_tokens: bool = True, **kwargs) -> List[int]:
        # Simple encode for now
        return self.tokenizer.encode(text)

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True, **kwargs) -> str:
        return self.tokenizer.decode(token_ids)

    def save_pretrained(self, save_directory: str):
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Save the core tokenizer
        self.tokenizer.save(os.path.join(save_directory, "spec_tokenizer.pkl"))

        # Save vocab for compatibility
        with open(os.path.join(save_directory, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

    @classmethod
    def from_pretrained(cls, load_directory: str):
        tokenizer = SpecTokenizer.load(os.path.join(load_directory, "spec_tokenizer.pkl"))
        return cls(tokenizer)
