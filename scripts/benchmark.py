import os
import sys
import time
import numpy as np
import tiktoken
from transformers import AutoTokenizer
from spec_bpe.tokenizer import SpecTokenizer

def calculate_fertility(tokens, text):
    words = text.split()
    if not words: return 0
    return len(tokens) / len(words)

def calculate_compression_ratio(tokens, text):
    bytes_len = len(text.encode('utf-8'))
    if not tokens: return 0
    return bytes_len / len(tokens)

def run_benchmark():
    # Sample text: A mix of technical and general content
    sample_text = """
    The transformer architecture has revolutionized natural language processing.
    By modeling global dependencies with self-attention, these models capture
    complex linguistic manifolds. Topology and spectral graph theory provide
    new insights into how information is crystallized in subword units.
    In the context of SPEC-BPE, we use Density Intelligence to identify stable
    semantic crystals rather than relying on simple frequency-driven heuristics.
    """ * 10

    print("Benchmarking Tokenizers...")
    print("-" * 40)

    # 1. SPEC-BPE
    spec_tokenizer = SpecTokenizer(vocab_size=1000)
    start = time.time()
    spec_tokenizer.train(sample_text)
    train_time = time.time() - start

    spec_tokens = spec_tokenizer.encode(sample_text)
    spec_cr = calculate_compression_ratio(spec_tokens, sample_text)
    spec_fer = calculate_fertility(spec_tokens, sample_text)

    print(f"SPEC-BPE (Vocab 1000):")
    print(f"  Training Time: {train_time:.2f}s")
    print(f"  Compression Ratio: {spec_cr:.2f} bytes/token")
    print(f"  Fertility: {spec_fer:.2f} tokens/word")
    print()

    # 2. Tiktoken (cl100k_base)
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        tik_tokens = enc.encode(sample_text)
        tik_cr = calculate_compression_ratio(tik_tokens, sample_text)
        tik_fer = calculate_fertility(tik_tokens, sample_text)
        print(f"Tiktoken (cl100k_base):")
        print(f"  Compression Ratio: {tik_cr:.2f} bytes/token")
        print(f"  Fertility: {tik_fer:.2f} tokens/word")
        print()
    except Exception as e:
        print(f"Tiktoken benchmark failed: {e}")

    # 3. Transformers GPT-2
    try:
        # Use token explicitly
        token = os.getenv("HF_TOKEN")
        gpt2_tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2", token=token)
        gpt2_tokens = gpt2_tokenizer.encode(sample_text)
        gpt2_cr = calculate_compression_ratio(gpt2_tokens, sample_text)
        gpt2_fer = calculate_fertility(gpt2_tokens, sample_text)
        print(f"GPT-2 (Transformers):")
        print(f"  Compression Ratio: {gpt2_cr:.2f} bytes/token")
        print(f"  Fertility: {gpt2_fer:.2f} tokens/word")
        print()
    except Exception as e:
        print(f"GPT-2 benchmark failed: {e}")

if __name__ == "__main__":
    run_benchmark()
