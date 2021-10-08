#!/usr/bin/env python

import argparse
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer, BertWordPieceTokenizer, SentencePieceBPETokenizer, CharBPETokenizer

special_tokens = [
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tokenizer', default="char-bpe")
    parser.add_argument('-i', '--input_path', default="./corpus/")
    parser.add_argument('-o', '--output', default="gptq")
    parser.add_argument('-s', '--vocab_size', type=int, default=12)
    parser.add_argument('-f', '--min_freq', type=int, default=2)
    args = parser.parse_args()

    if args.tokenizer == "wordpiece":
        tokenizer = BertWordPieceTokenizer()
    elif args.tokenizer == "byte-level-bpe":
        tokenizer = ByteLevelBPETokenizer()
    elif args.tokenizer == "sentencepiece-bpe":
        tokenizer = SentencePieceBPETokenizer()
    elif args.tokenizer == "char-bpe":
        tokenizer = CharBPETokenizer()
    else:
        raise RuntimeError(f"Uknown tokenizer {args.tokenizer}")

    paths = [str(x) for x in Path("./corpus/").glob("*.txt")]

    tokenizer.train(files=paths, vocab_size=args.vocab_size, min_frequency=args.min_freq, special_tokens=special_tokens)

    tokenizer.save_model(".", args.output)
