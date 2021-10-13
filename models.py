import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import RandomLayers


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
                    #    output[c][i, j, k] = q_result[c]
        return output


class FeedForward(pl.LightningModule):
    '''
    Used to create contextualized embeddings from the attention heads
    '''
    def __init__(self,
                 embed_dim,
                 boom_factor=4,
                 dropout=0.1,
                 kernel_size=5):
        super(FeedForward, self).__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd to conserve embedding dimension"
        padding = (kernel_size - 1) // 2

        self.c_fc = QConv1d(kernel_size, out_channels=boom_factor, padding=padding)

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
        return x
        #return self.dropout(self.c_proj(self.act(self.c_fc(x))))