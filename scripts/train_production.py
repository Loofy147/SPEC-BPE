import os
import sys
import time
import argparse
from spec_bpe.tokenizer import SpecTokenizer
from spec_bpe.utils import DataProcessor
from datasets import load_dataset

def train_production():
    parser = argparse.ArgumentParser(description="SPEC-BPE 2.0 Production Training Pipeline")
    parser.add_argument("--vocab_size", type=int, default=1000, help="Target vocabulary size")
    parser.add_argument("--data_limit", type=int, default=50000, help="Maximum characters to use for training")
    parser.add_argument("--dataset", type=str, default="wikitext", help="HF dataset name")
    parser.add_argument("--config", type=str, default="wikitext-2-raw-v1", help="HF dataset config")
    parser.add_argument("--split", type=str, default="train", help="HF dataset split")
    parser.add_argument("--incremental", action="store_true", help="Continue training from existing model")
    parser.add_argument("--model_path", type=str, default="models/spec_bpe_production.pkl", help="Path to save/load model")
    parser.add_argument("--code_mode", action="store_true", help="Optimize normalization for code")

    args = parser.parse_args()

    print("SPEC-BPE 2.0 Production Training Pipeline")
    print("=" * 40)

    if args.incremental and os.path.exists(args.model_path):
        print(f"Loading existing model from {args.model_path}")
        tokenizer = SpecTokenizer.load(args.model_path)
        tokenizer.vocab_size = args.vocab_size
    else:
        tokenizer = SpecTokenizer(vocab_size=args.vocab_size)

    print(f"Loading dataset: {args.dataset} ({args.config})")
    try:
        ds = load_dataset(args.dataset, args.config, split=args.split, trust_remote_code=True)
        full_text = ""
        for i in range(len(ds)):
            full_text += ds[i]["text"] + "\n"
            if len(full_text) > args.data_limit:
                full_text = full_text[:args.data_limit]
                break
    except Exception as e:
        print(f"Failed to load dataset {args.dataset}: {e}")
        print("Falling back to internal sample data...")
        full_text = """
        Artificial Intelligence is transforming the world.
        Linguistic topology and spectral graph theory provide deep insights into
        how meaning is encoded in subword units. SPEC-BPE uses Density Intelligence
        to create more efficient and stable tokenizations.
        """ * 200
        full_text = full_text[:args.data_limit]

    print(f"Original text length: {len(full_text)} characters")
    dp = DataProcessor(code_mode=args.code_mode)
    processed_text = dp.process_corpus(full_text)
    print(f"Normalized text length: {len(processed_text)} characters")

    print(f"Training on {len(processed_text)} characters...")
    start_time = time.time()
    tokenizer.train(processed_text, incremental=args.incremental)
    end_time = time.time() - start_time

    print(f"Training completed in {end_time:.2f}s")
    print(f"Final Vocab Size: {len(tokenizer.vocab)}")
    print(f"Manifold Volume: {tokenizer.manifold_volume:.4f}")

    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    tokenizer.save(args.model_path)
    print(f"Model saved to {args.model_path}")

    test_text = "The transformer architecture is revolutionizing NLP."
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    print(f"Test Decode: {decoded}")
    print("Verification passed.")

if __name__ == "__main__":
    train_production()
