import math
import copy

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn import ConstantPad1d
from torch.nn.modules import ModuleList
from torch.nn.modules.normalization import LayerNorm

import lightning as L
from torchmetrics.functional.classification.accuracy import accuracy
from transformers import PreTrainedTokenizerFast

import pennylane as qml
from pennylane import numpy as np
#from pennylane.templates import RandomLayers

from utils import pad_sequence


class QConv1d(L.LightningModule):
    '''
    out_sz = (input_sz + 2*padding - kernel_sz) / stride + 1
    '''
    def __init__(self,
                 kernel_size,
                 out_channels=3,  # ie Query, Key, Value
                 n_qlayers=1,
                 q_device='lightning.qubit',
                 stride=1,
                 padding=0,
                 **kwargs):
        super().__init__()
        self.out_channels = out_channels
        self.padding = padding
        self.stride = stride
        self.kernel_size = kernel_size
        self.n_qlayers = n_qlayers
        assert self.kernel_size >= self.out_channels
        self.weights = np.random.uniform(high= 2 * np.pi, size=(self.n_qlayers, self.kernel_size))
        dparams = {}
        if q_device in ["braket.aws.qubit"]:
            dparams['device_arn'] = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"
            dparams['s3_destination_folder'] = ("amazon-braket-ideal-datasets", "gptq")
        self.dev = qml.device(q_device, wires=self.kernel_size, **dparams)

        @qml.qnode(device=self.dev, interface="torch")
        def _circuit(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=range(self.kernel_size))
            qml.templates.BasicEntanglerLayers(weights, wires=range(self.kernel_size))
            # NB: you may leave some qubits out
            return [qml.expval(qml.PauliZ(j)) for j in range(self.out_channels)]

        weight_shapes = {"weights": (self.n_qlayers, self.kernel_size)}
        self.qconv = qml.qnn.TorchLayer(_circuit, weight_shapes)
    
    def draw(self):
        # build circuit by sending dummy data through it
        _ = self.qconv(inputs=torch.from_numpy(np.zeros(self.kernel_size)))
        dummy_inputs = np.random.random(self.kernel_size)
        dummy_weights = np.random.random((self.n_qlayers, self.kernel_size))
        print(qml.draw(self.qconv.qnode)(dummy_inputs, dummy_weights))
        self.qconv.zero_grad()

    def forward(self, x):
        embed_dim = x.shape[-1]
        x = F.pad(x, (self.padding, self.padding), "constant", 0)
        out_dim = int((embed_dim + 2 * self.padding - self.kernel_size) / self.stride) + 1

        # the code below creates indices for a sliding window of size kernel_size 
        idx = torch.unsqueeze(torch.arange(self.kernel_size), 0) + torch.unsqueeze(torch.arange(out_dim) * self.stride, 0).T
        x = x[:, :, idx]
        return self.qconv(x)
        '''
        batch_size, seq_len, embed_dim = x.shape
        output = torch.zeros((batch_size, seq_len, out_dim, self.out_channels))
        for i in range(batch_size):
            for j in range(seq_len):
                for k in range(0, out_dim):
                    k_start = k*self.stride
                    k_end = k_start + self.kernel_size
                    x_slice  = x[i, j, k_start:k_end]
                    q_result = self.qconv(x_slice)
                    for c in range(self.out_channels):
                        output[i, j, k, c] = q_result[c]
        return output
        '''


class FeedForwardQuantum(L.LightningModule):
    '''
    Used to create contextualized embeddings from the attention heads
    '''
    def __init__(self,
                 embed_dim,
                 boom_factor=4,
                 dropout_rate=0.1,
                 n_qubits: int=5,
                 n_qlayers: int=1,
                 q_device: str="lightning.qubit",
                 **kwargs):
        super().__init__()
        assert n_qubits % 2 == 1, "Kernel size must be odd to conserve embedding dimension"
        padding = (n_qubits - 1) // 2

        self.c_fc = QConv1d(kernel_size=n_qubits,
                            out_channels=boom_factor,
                            padding=padding,
                            n_qlayers=n_qlayers,
                            q_device=q_device)

        s_inv = (boom_factor * embed_dim - boom_factor) // (embed_dim - 1)
        self.c_proj = QConv1d(kernel_size=boom_factor,
                              out_channels=1,
                              stride=s_inv,
                              n_qlayers=n_qlayers,
                              q_device=q_device)
        self.activation = F.gelu
        self.dropout = nn.Dropout(dropout_rate)
 
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


class MultiHeadAttentionQuantum(L.LightningModule):
    def __init__(self,
                 embed_dim: int=8,
                 n_heads: int=2,
                 n_qubits: int=5,
                 n_qlayers: int=1,
                 q_device: str="lightning.qubit",
                 **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        padding = (n_qubits - 1) // 2  # does not change embed_dim
        self.c_attn = QConv1d(kernel_size=n_qubits,
                              out_channels=3,
                              n_qlayers=n_qlayers,
                              padding=padding,
                              q_device=q_device)
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
        q, k, v = x[:, :, :,  0], x[:, :, :, 1], x[:, :, :, 2]
        q, k, v = self.split_heads(q), self.split_heads(k), self.split_heads(v)
        out = self._attn(q, k, v, mask)
        out = self.merge_heads(out)
        out = self.c_proj(out)
        out = torch.squeeze(out, dim=-1)
        return out


class TransformerBlockQuantum(L.LightningModule):
    def __init__(self,
                 embed_dim,
                 n_heads: int=2,
                 n_qubits: int=5,
                 n_qlayers: int=1,
                 q_device: str="lightning.qubit",
                 dropout_rate=0.1,
                 **kwargs):
        super().__init__()
        self.attn = MultiHeadAttentionQuantum(embed_dim,
                              n_heads=n_heads,
                              n_qubits=n_qubits,
                              n_qlayers=n_qlayers,
                              q_device=q_device)
        self.feedforward = FeedForwardQuantum(embed_dim,
                                       boom_factor=4,
                                       dropout_rate=dropout_rate,
                                       n_qubits=n_qubits,
                                       n_qlayers=n_qlayers,
                                       q_device=q_device)
        self.ln_1 = LayerNorm(embed_dim)
        self.ln_2 = LayerNorm(embed_dim)

    def forward(self, x, src_mask=None):
        x = x + self.attn(self.ln_1(x), src_mask)
        x = x + self.feedforward(self.ln_2(x))
        return x


class GPTBase(L.LightningModule):
    '''
    If you don't like C++ multiple inheritance, try Python
    '''
    def __init__(self,
                 embed_dim: int,
                 tgt_vocab: int,
                 n_heads: int=4,
                 dropout_rate=0.1,
                 n_tlayers: int=1,
                 max_seq_len: int=512,
                 tokenizer_file: str="gptq.json",
                 **kwargs):
        super().__init__()
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
        self.embed_dim = embed_dim
        self.src_vocab = self.tokenizer.vocab_size #src_vocab
        self.tgt_vocab = tgt_vocab
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate
        self.n_tlayers = n_tlayers
        self.max_seq_len = max_seq_len
        self.h = None
        self.wte = nn.Embedding(self.src_vocab, self.embed_dim)
        self.wpe = nn.Embedding(self.max_seq_len, self.embed_dim)  # this is learned, not pre-computed
        self.dropout = nn.Dropout(self.dropout_rate)
        self.ln_f = LayerNorm(self.embed_dim)
        self.attn_mask = self.generate_square_subsequent_mask(self.max_seq_len)
        self.init_weights()

    def get_word_embedding_dimension(self):
        return self.embed_dim
    
    def tokenize(self, X):
        return self.tokenizer(X)

    @staticmethod
    def generate_square_subsequent_mask(sz: int) -> Tensor:
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

    def _create_tranformer_layers(self):
        raise NotImplementedError("GPT base class cannot create transformer layers")

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, inputs, src_mask=None):
        token_ids = [torch.LongTensor(x) for x in inputs['input_ids']]
        token_ids = pad_sequence(token_ids, self.max_seq_len)
        # token_type_ids = inputs['token_type_ids']
        # attention_mask = inputs['attention_mask']
        if src_mask is None:
            src_mask = self.attn_mask
        pos_ids = torch.arange(0, token_ids.size(-1)).unsqueeze(0)
        x_tokens = self.wte(token_ids)
        x_pos = self.wpe(pos_ids)
        x = x_tokens + x_pos
        x = self.dropout(x)
        for i in range(self.n_tlayers): 
            x = self.h[i](x, src_mask)
        x = self.ln_f(x)
        return x


class GPT2(GPTBase):
    def __init__(self, 
                 embed_dim,
                 tgt_vocab,
                 **kwargs):
        super().__init__(embed_dim, tgt_vocab, **kwargs)
        self._create_tranformer_layers()
        self.out = nn.Linear(embed_dim, tgt_vocab, bias=False)
        self.init_weights()

    def _create_tranformer_layers(self):
        encoder_template = nn.TransformerEncoderLayer(d_model=self.embed_dim,
                                                      nhead=self.n_heads,
                                                      dropout=self.dropout_rate,
                                                      dim_feedforward=4*self.embed_dim,
                                                      batch_first=True
                            )
        self.h = ModuleList([
            nn.TransformerEncoder(encoder_template, self.n_tlayers) for _ in range(self.n_tlayers)
        ])


class GPTQ(GPTBase):
    def __init__(self,
                 embed_dim,
                 tgt_vocab,
                 n_qlayers: int=1,
                 q_device: str="lightning.qubit",  # lightning.gpu, braket.aws.qubit, default.qubit
                 **kwargs):
        super().__init__(embed_dim, tgt_vocab, **kwargs)
        self.n_qlayers = n_qlayers
        self.q_device = q_device
        self._create_tranformer_layers()
        self.out = nn.Linear(embed_dim, tgt_vocab, bias=False)  # quantum-fy this, too?
        # out_sz = (input_sz + 2*padding - kernel_sz) / stride + 1
        # tgt_vocab = (embed_dim + 2*padding - kernel_sz) / stride + 1
        # self.out = QConv1d(kernel_size=, out_channels=, n_qlayers=self.n_qlayers, stride=stride, padding=padding)
        self.init_weights()

    def _create_tranformer_layers(self):
        self.h = ModuleList([
            TransformerBlockQuantum(embed_dim=self.embed_dim,
                                    n_heads=self.n_heads,
                                    dropout_rate=self.dropout_rate,
                                    n_qlayers=self.n_qlayers,
                                    q_device=self.q_device) for _ in range(self.n_tlayers)
        ])


class IMDbClassifierBase(L.LightningModule):
    def __init__(self, lr=1e-3, **kwargs):
        self.lr = lr

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        probs = F.log_softmax(logits, dim=-1)
        loss = F.nll_loss(probs, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
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


class IMDbClassifier(IMDbClassifierBase, GPT2):
    def __init__(self,
                 embed_dim,
                 # vocab_size: int=2000,
                 n_heads: int=4,
                 dropout_rate: float=0.1,
                 n_tlayers: int=1,
                 max_seq_len: int=1024,
                 lr=1e-3):
        IMDbClassifierBase.__init__(self, lr=lr)
        GPT2.__init__(self,
            embed_dim=embed_dim,
            # src_vocab=vocab_size,
            tgt_vocab=2,
            n_heads=n_heads,
            dropout_rate=dropout_rate,
            n_tlayers=n_tlayers,
            max_seq_len=max_seq_len)

    def forward(self, token_ids, src_mask=None):
        x = GPT2.forward(self, token_ids, src_mask)
        x = x.mean(dim=1)  # average across tokens for each embedding dim
        logits = self.out(x)
        return logits


class IMDbClassifierQuantum(IMDbClassifierBase, GPTQ):
    def __init__(self,
                 embed_dim,
                 # vocab_size: int=2000,
                 n_heads: int=4,
                 dropout_rate: float=0.1,
                 n_tlayers: int=1,
                 max_seq_len: int=1024,
                 n_qlayers: int=1,
                 q_device: str="lightning.qubit",
                 lr=1e-3):
        IMDbClassifierBase.__init__(self, lr=lr)
        GPTQ.__init__(self,
            embed_dim=embed_dim,
            # src_vocab=vocab_size,
            tgt_vocab=2,
            n_heads=n_heads,
            dropout_rate=dropout_rate,
            n_tlayers=n_tlayers,
            max_seq_len=max_seq_len,
            n_qlayers=n_qlayers,
            q_device=q_device)

    def forward(self, token_ids, src_mask=None):
        x = GPTQ.forward(self, token_ids, src_mask)
        x = x.mean(dim=1)  # average across tokens for each embedding dim
        logits = self.out(x)
        return logits


class LanguageModel(GPTQ):
    def __init__(self, 
                 embed_dim,
                 # vocab_size: int=2000,
                 n_heads: int=4,
                 dropout=0.1,
                 n_layers: int=1,
                 max_seq_len: int=1024,
                 lr=1e-3):
        self.lr = lr
        super().__init__(
            embed_dim=embed_dim,
            # src_vocab=vocab_size,
            tgt_vocab=vocab_size,
            n_heads=n_heads,
            dropout=dropout,
            n_layers=n_layers,
            max_seq_len=max_seq_len)

    def forward(self, x, src_mask=None):
        x = super().forward(x, src_mask)
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