import torch

from pathlib import Path

from torch.utils.data import Dataset
from tokenizers import ByteLevelBPETokenizer, BertWordPieceTokenizer, SentencePieceBPETokenizer, CharBPETokenizer
from tokenizers.processors import BertProcessing


class GPTQDataset(Dataset):
    def __init__(self,
                 evaluate=False,
                 max_length=512,
                 tokenizer="char-bpe"
                ):
        if tokenizer == "wordpiece":
            self.tokenizer = BertWordPieceTokenizer("./gptq-vocab.txt")
        elif tokenizer == "byte-level-bpe":
            self.tokenizer = ByteLevelBPETokenizer("./gptq-vocab.json", "./gptq-merges.txt")
        elif tokenizer == "sentencepiece-bpe":
            self.tokenizer = SentencePieceBPETokenizer("./gptq-vocab.json", "./gptq-merges.txt")
        elif tokenizer == "char-bpe":
            self.tokenizer = CharBPETokenizer("./gptq-vocab.json", "./gptq-merges.txt")
        else:
            raise RuntimeError(f"Uknown tokenizer {tokenizer}")

        self.tokenizer._tokenizer.post_processor = BertProcessing(
            ("</s>", tokenizer.token_to_id("</s>")),
            ("<s>", tokenizer.token_to_id("<s>")),
        )
        self.tokenizer.enable_truncation(max_length=max_length)

        self.examples = []

        src_files = Path("./data/").glob("*-eval.txt") if evaluate else Path("./data/").glob("*-train.txt")
        for src_file in src_files:
            print("🔥", src_file)
            lines = src_file.read_text(encoding="utf-8").splitlines()
            self.examples += [x.ids for x in self.tokenizer.encode_batch(lines)]

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, i):
            # We’ll pad at the batch level.
            return torch.tensor(self.examples[i])