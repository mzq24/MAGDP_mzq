import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=11):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:,:]
        return self.dropout(x)

class TransformerModel_m(nn.Module):
    def __init__(self, input_dim, emb_dim, output_dim, num_layers=2, num_heads=8, preprocess_type=1):
        '''
        preprocess_type: 0 for flat, 1 for pooling
        '''
        super(TransformerModel_m, self).__init__()
        self.embedding = nn.Linear(input_dim, emb_dim)
        self.preprocess_type = preprocess_type
        self.position_encoding = PositionalEncoding(d_model=emb_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads, batch_first=True),
            num_layers
        )
        self.fc = nn.Linear(emb_dim, output_dim)
        
    def forward(self, x: torch.Tensor):
        x = self.embedding(x)
        x = self.position_encoding(x)
        x = self.transformer_encoder(x)
        if self.preprocess_type == 0:
            x = x.view(x.size(0), -1)  # Flatten the output
        elif self.preprocess_type == 1:
            x = x.mean(dim=1, keepdim=True).squeeze(1)   # Pooling
        x = self.fc(x)
        return x

if __name__ == '__main__':
    # 创建一个Transformer模型实例
    input_dim = 6
    hidden_dim = 256
    num_layers = 4
    num_heads = 8
    model = TransformerModel_m(input_dim, hidden_dim, num_layers, num_heads)

    # 输入数据
    batch_size = 32
    seq_len = 11
    input_data = torch.randn(batch_size, seq_len, input_dim)

    # 前向传播
    output = model(input_data)
