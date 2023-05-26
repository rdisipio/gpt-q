#!/usr/bin/env python

import csv
import math
import os
import gzip
import torch

import numpy as np

from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, models, losses, util, InputExample, LoggingHandler
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from tokenizers import ByteLevelBPETokenizer, BertWordPieceTokenizer, SentencePieceBPETokenizer, CharBPETokenizer

from models import GPTQ

embed_dim = 8
vocab_size = 512
output_features = 8
n_heads = 4
dropout_rate = 0.1
n_tlayers = 1
max_seq_len = 16
n_qlayers = 1
n_qubits = 5 # must be odd and > 3 (ie query, key, value)
#q_device = "lightning.qubit" # lightning.gpu, braket.aws.qubit, default.qubit
q_device = "qulacs.simulator"
#q_device = "braket.aws.qubit"
lr = 1e-3

model_name = 'gptq'
train_batch_size = 16
num_epochs = 2
model_save_path = 'output/training_stsbenchmark_continue_training-'+model_name+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

special_tokens = [
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ]

train_samples = []
dev_samples = []
test_samples = []
sts_dataset_path = 'datasets/stsbenchmark.tsv.gz'
with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1
        inp_example = InputExample(texts=[row['sentence1'], row['sentence2']], label=score)

        if row['split'] == 'dev':
            dev_samples.append(inp_example)
        elif row['split'] == 'test':
            test_samples.append(inp_example)
        else:
            train_samples.append(inp_example)

# hack
n = 5
train_samples = train_samples[:n]
test_samples = test_samples[:n]


gptq = GPTQ(embed_dim=embed_dim,
            tgt_vocab=vocab_size,
            n_heads=n_heads,
            dropout_rate=dropout_rate,
            n_tlayers=n_tlayers,
            max_seq_len=max_seq_len,
            n_qlayers=n_qlayers,
            n_qubits=n_qubits,
            q_device=q_device,
            batch_first=True)
pooling_model = models.Pooling(gptq.get_word_embedding_dimension())
dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=output_features, activation_function=torch.nn.Tanh())

model = SentenceTransformer(modules=[gptq, pooling_model, dense_model])

train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
train_loss = losses.CosineSimilarityLoss(model=model)

evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-dev')


#warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
warmup_steps = 1
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=1,
          warmup_steps=warmup_steps,
          output_path=model_save_path)
