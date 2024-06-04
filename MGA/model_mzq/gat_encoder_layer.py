import torch.nn as nn
from torch.nn import Linear, Dropout, LeakyReLU
from torch.nn import LayerNorm


class GATEncoderLayer(nn.Module):
    def __init__(self, gnn_conv: int, dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5,
                 bias: bool = True):
        super().__init__()
        self.gnn_conv = gnn_conv
        # d_model = gnn_conv.out_channels
        d_norm  = gnn_conv.out_channels * gnn_conv.heads
        dim_feedforward = 4 * d_norm
        # Implementation of Feedforward model
        self.linear1 = Linear(d_norm, dim_feedforward, bias=bias)
        self.linear2 = Linear(dim_feedforward, d_norm, bias=bias)

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.norm1 = LayerNorm(d_norm, eps=layer_norm_eps)
        self.norm2 = LayerNorm(d_norm, eps=layer_norm_eps)

        self.activation = LeakyReLU(0.1)

    def forward(self, x, edges):
        # print(f'x {x.shape}')
        # print(self.gnn_conv)
        # print(x.isnan().any())
        x = self.norm1(x + self.activation(self.gnn_conv(x, edges)))
        # x = self.norm1(x + self.gnn_conv(x, edges))
        # print(x.isnan().any())
        x = self.norm2(x + self._ff_block(x))

        return x
    
    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout1(self.activation(self.linear1(x))))
        return self.dropout2(x)