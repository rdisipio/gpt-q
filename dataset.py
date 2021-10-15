import torch

from pathlib import Path

from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.model_selection import train_test_split

from tokenizers import ByteLevelBPETokenizer, BertWordPieceTokenizer, SentencePieceBPETokenizer, CharBPETokenizer
from tokenizers.processors import BertProcessing

from torchtext.datasets import IMDB
from pytorch_lightning import LightningDataModule


class IMDbData(Dataset):
    def __init__(self,
                 data_iter,
                 max_length=512,
                 tokenizer="char-bpe"
                ):
        super(IMDbData, self).__init__()

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
        self.labels = []
        for label, line in data_iter:
            self.examples.append(line)
            self.labels.append(label)
        self.examples = [x.ids for x in self.tokenizer.encode_batch(self.examples)]

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, i):
            token_ids = self.examples[i]
            return torch.tensor(token_ids), torch.tensor([label])


class IMDbDataModule(LightningDataModule):
    def __init__(self, val_split=0.2, batch_size=32, max_seq_length=512, tokenizer="char-bpe"):
        super(IMDbDataModule, self).__init__()

        self.val_split = val_split
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length

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
        
        self.special_tokens = [
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<mask>",
        ]

        self.tokenizer._tokenizer.post_processor = BertProcessing(
            ("</s>", self.tokenizer.token_to_id("</s>")),
            ("<s>", self.tokenizer.token_to_id("<s>")),
        )
        self.tokenizer.enable_truncation(max_length=max_seq_length)
        self.tokenizer.enable_padding(pad_id=1, pad_token="<pad>")

    def _pad(self, x):
        n = len(x)
        if n >= self.max_seq_length:
            return x
        return x + [0] * (self.max_seq_length - n)

    def _review_to_id(self, y):
        return [1 if a == 'pos' else 0 for a in y]

    def tokenize(self, data_iter):
        X = []
        y = []
        for label, line in data_iter:
            X.append(line)
            y.append(label)
        X = [self._pad(x.ids) for x in self.tokenizer.encode_batch(X)]
        y = self._review_to_id(y)

        X = [torch.LongTensor(x) for x in X]
        #X = torch.Tensor(X)
        #y = torch.Tensor(y)
        return [z for z in zip(X, y)]

    def prepare_data(self):
        '''Download data'''
        train_iter, test_iter = IMDB(split=('train', 'test'))

        self.train_data = self.tokenize(train_iter)
        self.imdb_test = self.tokenize(test_iter)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            train_idx, val_idx = train_test_split(list(range(len(self.train_data))), test_size=self.val_split)

            self.imdb_train = Subset(self.train_data, train_idx)
            self.imdb_val = Subset(self.train_data, val_idx)
        #if stage == "test" or stage is None:
        #    self.imdb_test = None

    def train_dataloader(self):
        return DataLoader(self.imdb_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.imdb_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.imdb_test, batch_size=self.batch_size)
