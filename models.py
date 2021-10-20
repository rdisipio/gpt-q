import math
import copy

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules import ModuleList
from torch.nn.modules.normalization import LayerNorm

import pytorch_lightning as pl
from torchmetrics.functional.classification.accuracy import accuracy

import pennylane as qml
from pennylane import numpy as np
#from pennylane.templates import RandomLayers

from utils import make_src_mask


class QConv1d(pl.LightningModule):
    def __init__(self,
                 kernel_size,
                 out_channels=3,  # ie Query, Key, Value
                 n_qlayers=1,
                 q_device='default.qubit',
                 stride=1,
                 padding=0):
        super(QConv1d, self).__init__()
        self.out_channels = out_channels
        self.padding = padding
        self.stride = stride
        self.kernel_size = kernel_size
        assert self.kernel_size >= self.out_channels
        self.dev = qml.device(q_device, wires=self.kernel_size)
        self.weights = np.random.uniform(high= 2 * np.pi, size=(n_qlayers, self.kernel_size))

        @qml.qnode(device=self.dev, interface="torch")
        def _circuit(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=range(self.kernel_size))
            qml.templates.BasicEntanglerLayers(weights, wires=range(self.kernel_size))
            # NB: you may leave some qubits out
            return [qml.expval(qml.PauliZ(j)) for j in range(self.out_channels)]

        weight_shapes = {"weights": (n_qlayers, self.kernel_size)}
        self.qconv = qml.qnn.TorchLayer(_circuit, weight_shapes)
    
    def draw(self):
        # build circuit by sending dummy data through it
        _ = self.qconv(inputs=torch.from_numpy(np.zeros(self.kernel_size)))
        print(self.qconv.qnode.draw())
        self.qconv.zero_grad()

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        x = F.pad(x, (self.padding, self.padding), "constant", 0)
        out_dim = int((embed_dim + 2 * self.padding - self.kernel_size) / self.stride) + 1
        output = torch.zeros((batch_size, seq_len, out_dim, self.out_channels))
        for i in range(batch_size):
            for j in range(seq_len):
                for k in range(0, out_dim, self.stride):
                    k_end = k + self.kernel_size
                    x_slice  = x[i, j, k:k_end]
                    q_result = self.qconv(x_slice)
                    for c in range(self.out_channels):
                        output[i, j, k, c] = q_result[c]
        return output


class FeedForward(pl.LightningModule):
    '''
    Used to create contextualized embeddings from the attention heads
    '''
    def __init__(self,
                 embed_dim,
                 boom_factor=4,
                 dropout=0.1,
                 n_qubits: int=5,
                 n_qlayers: int=1):
        super(FeedForward, self).__init__()
        assert n_qubits % 2 == 1, "Kernel size must be odd to conserve embedding dimension"
        padding = (n_qubits - 1) // 2

        self.c_fc = QConv1d(kernel_size=n_qubits, out_channels=boom_factor, padding=padding, n_qlayers=n_qlayers)

        s_inv = (boom_factor * embed_dim - boom_factor) // (embed_dim - 1)
        self.c_proj = QConv1d(kernel_size=boom_factor, out_channels=1, stride=s_inv)
        self.activation = F.gelu
        self.dropout = nn.Dropout(dropout)
 
    def _scatter_and_merge(self, x):
        batch_sz, seq_len, embed_dim, n_ch = x.shape
        x = x.transpose(-1,-2)
        x = x.view(batch_sz, seq_len, n_ch, -1)
        x = x.reshape((batch_sz, seq_len, n_ch * embed_dim))
        return x

    def forward(self, x):
        x = self.c_fc(x)
        x = self._scatter_and_merge(x)
        x = self.activation(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return torch.squeeze(x, dim=-1)


class MultiHeadAttention(pl.LightningModule):
    def __init__(self,
                 embed_dim: int=8,
                 n_heads: int=2,
                 n_qubits: int=5,
                 n_qlayers: int=1,
                 ):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        padding = (n_qubits - 1) // 2  # does not change embed_dim
        self.c_attn = QConv1d(kernel_size=n_qubits,
                              out_channels=3,
                              n_qlayers=n_qlayers,
                              padding=padding)
        self.softmax = nn.Softmax(dim=-1)
        #self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.dropout = nn.Dropout(0.1)
        self.c_proj = QConv1d(kernel_size=n_qubits,
                              out_channels=1,
                              padding=padding)

    def split_heads(self, x):
        new_shape = x.size()[:-1] + (self.n_heads, x.size(-1) // self.n_heads)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3) 

    def _attn(self, q, k, v, attn_mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1))
        sf = math.sqrt(v.size(-1))
        scores = scores / sf
        #nd, ns  = scores.size(-2), scores.size(-1)
        if attn_mask is not None:
            # we add float('-inf') to tokens we want to suppress
            # so the softmax prob is 0
            attn_mask = attn_mask.unsqueeze(0)
            scores += attn_mask
            #scores = scores.masked_fill(attn_mask == 0, -1e9)
            #scores = scores.float().masked_fill(attn_mask, -float('inf')).type_as(scores)
        scores  = self.softmax(scores)
        scores  = self.dropout(scores)
        outputs = torch.matmul(scores, v)
        return outputs

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_shape)

    def forward(self, x, mask=None):
        x = self.c_attn(x)
        #print("after qconv 1->3:", x.shape)
        q, k, v = x[:, :, :,  0], x[:, :, :, 1], x[:, :, :, 2]
        #print("shapes: q:", q.shape, "k:", k.shape, "v:", v.shape)
        q, k, v  = self.split_heads(q), self.split_heads(k), self.split_heads(v)
        #print("after split heads:", q.shape, k.shape, v.shape)
        out      = self._attn(q, k, v, mask)
        #print("attention:", out.shape)
        out      = self.merge_heads(out)
        #print("merged heads:", out.shape)
        out      = self.c_proj(out)
        out = torch.squeeze(out, dim=-1)
        #print("attn output:", out.shape)
        return out


class TransformerBlock(pl.LightningModule):
    def __init__(self,
                 embed_dim,
                 n_heads: int=2,
                 n_qubits: int=5,
                 n_qlayers: int=1,
                 dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attn = MultiHeadAttention(embed_dim,
                              n_heads=n_heads,
                              n_qubits=n_qubits,
                              n_qlayers=n_qlayers)
        self.feedforward = FeedForward(embed_dim,
                                       boom_factor=4,
                                       dropout=dropout,
                                       n_qubits=n_qubits,
                                       n_qlayers=n_qlayers)
        self.ln_1 = LayerNorm(embed_dim)
        self.ln_2 = LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln_1(x), mask)
        x = x + self.feedforward(self.ln_2(x))
        return x


class GPTQ(pl.LightningModule):
    def __init__(self,
                 embed_dim,
                 src_vocab,
                 tgt_vocab,
                 n_heads: int=4,
                 dropout=0.1,
                 n_layers: int=1,
                 max_seq_len: int=1024,
                 ):
        super(GPTQ, self).__init__()
        self.n_layers = n_layers
        tblock = TransformerBlock(embed_dim, n_heads=n_heads, dropout=dropout)
        self.h = ModuleList([copy.deepcopy(tblock) for i in range(self.n_layers)])
        self.wte = nn.Embedding(src_vocab, embed_dim)
        self.wpe = nn.Embedding(max_seq_len, embed_dim)  # this is learned, not pre-computed
        self.dropout = nn.Dropout(dropout)
        self.ln_f    = LayerNorm(embed_dim)
        self.out     = nn.Linear(embed_dim, tgt_vocab, bias=False)  #QCNN, VQC?
        self.loss_fn = nn.CrossEntropyLoss()
        self.init_weights()
        self.attn_mask = make_src_mask(max_seq_len)

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _step(self, token_ids, mask=None):
        if mask is None:
            mask = self.attn_mask
        pos_ids = torch.arange(0, token_ids.size(-1)).unsqueeze(0)
        x_tokens = self.wte(token_ids)
        x_pos = self.wpe(pos_ids)
        x = x_tokens + x_pos
        x = self.dropout(x)
        for i in range(self.n_layers): 
            x = self.h[i](x, mask)
        x = self.ln_f(x)
        return x

    def forward(self, token_ids, mask=None):
        x = self._step(token_ids, mask)
        logits = self.out(x)
        return logits
        '''
        x = self._step1(src_ids, pos_ids)
        x = x.mean(dim=1)
        #print("output of transformer blocks:", x.shape)
        logits = self.out(x)
        #print("logits:", logits.shape)
        outputs = (logits,) + (x,)

        if tgt_ids is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = tgt_ids[..., 1:].contiguous()
            loss = self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            outputs = (loss,) + outputs
            return outputs
        return logits
        '''


class IMDbClassifier(GPTQ):
    def __init__(self,
                 embed_dim,
                 vocab_size: int=2000,
                 n_heads: int=4,
                 dropout=0.1,
                 n_layers: int=1,
                 max_seq_len: int=1024,
                 lr=1e-3):
        self.n_classes = 2
        self.lr = lr
        super(IMDbClassifier, self).__init__(
            embed_dim=embed_dim,
            src_vocab=vocab_size,
            tgt_vocab=self.n_classes,
            n_heads=n_heads,
            dropout=dropout,
            n_layers=n_layers,
            max_seq_len=max_seq_len)

    def forward(self, token_ids, mask=None):
        x = self._step(token_ids, mask)
        x = x.mean(dim=1)  # average across tokens for each embedding dim
        logits = self.out(x)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        probs = F.log_softmax(logits, dim=-1)
        loss = F.nll_loss(probs, y)
        #shift_logits = logits[..., :-1, :].contiguous()
        #shift_labels = tgt_ids[..., 1:].contiguous()
        #loss = F.nll_loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        probs = F.log_softmax(logits, dim=-1)
        loss = F.nll_loss(probs, y)
        preds = torch.argmax(probs, dim=-1)
        acc = accuracy(preds, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class LanguageModel(GPTQ):
    def __init__(self, 
                 embed_dim,
                 vocab_size: int=2000,
                 n_heads: int=4,
                 dropout=0.1,
                 n_layers: int=1,
                 max_seq_len: int=1024,
                 lr=1e-3):
        self.lr = lr
        super(LanguageModel, self).__init__(
            embed_dim=embed_dim,
            src_vocab=vocab_size,
            tgt_vocab=vocab_size,
            n_heads=n_heads,
            dropout=dropout,
            n_layers=n_layers,
            max_seq_len=max_seq_len)

    def forward(self, x, mask=None):
        x = self._step(x, mask=mask)
        logits = self.out(x)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        probs = F.log_softmax(logits, dim=-1)
        loss = F.nll_loss(probs, y)
        #shift_logits = logits[..., :-1, :].contiguous()
        #shift_labels = tgt_ids[..., 1:].contiguous()
        #loss = F.nll_loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        probs = F.log_softmax(logits, dim=-1)
        loss = F.nll_loss(probs, y)
        preds = torch.argmax(probs, dim=-1)
        acc = accuracy(preds, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer