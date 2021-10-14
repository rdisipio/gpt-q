#!/usr/bin/env python

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.metrics.functional import accuracy

from models import GPTQ, IMDbClassifier
from dataset import IMDbData



if __name__ == '__main__':
    AVAIL_GPUS = min(1, torch.cuda.device_count())
    BATCH_SIZE = 256 if AVAIL_GPUS else 32

    gptq = GPTQ()
    model = IMDbClassifier(gptq)

    trainer = Trainer(
        max_epochs=2,
        gpus=AVAIL_GPUS,
        progress_bar_refresh_rate=20)
    trainer.fit(model)