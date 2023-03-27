#!/usr/bin/env python

import argparse
import csv
import gzip
import os
from pathlib import Path
#from tokenizers import ByteLevelBPETokenizer, BertWordPieceTokenizer, SentencePieceBPETokenizer, CharBPETokenizer
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, trainers
from sentence_transformers import util

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
    parser.add_argument('-i', '--input_path', default="./datasets/")
    parser.add_argument('-o', '--output', default="gptq")
    parser.add_argument('-s', '--vocab_size', type=int, default=512)
    parser.add_argument('-f', '--min_freq', type=int, default=2)
    args = parser.parse_args()

    sts_dataset_path = 'datasets/stsbenchmark.tsv.gz'
    if not os.path.exists(sts_dataset_path):
        util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)
    
    train_data = []
    with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            train_data.extend([row['sentence1'], row['sentence2']])

    #with open("datasets/sentences.txt", 'w') as f:
    #    f.write("\n".join(sentences))

    #paths = [str(x) for x in Path("./corpus/").glob("*.txt")]
    #paths = ["datasets/sentences.txt"]
    
    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    tokenizer.decoders = decoders.ByteLevel()
    
    #tokenizer.train(files=paths, vocab_size=args.vocab_size, min_frequency=args.min_freq, special_tokens=special_tokens)
    #tokenizer.save_model(".", args.output)
    
    trainer = trainers.BpeTrainer(vocab_size=args.vocab_size,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  min_frequency=args.min_freq,
                                  special_tokens=special_tokens)
    tokenizer.train_from_iterator(train_data, trainer=trainer)
    tokenizer.save("gptq.json")
