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
        output = [torch.zeros((batch_size, seq_len, out_dim)) for _ in range(self.out_channels)]
        
        for i in range(batch_size):
            for j in range(seq_len):
                for k in range(0, out_dim, self.stride):
                    k_end = k + self.kernel_size
                    x_slice  = x[i, j, k:k_end]
                    q_result = self.qconv(x_slice)
                    for c in range(self.out_channels):
                        output[c][i, j, k] = q_result[c]
        return output

