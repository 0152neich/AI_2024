import torch
from torch import nn

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        # Khởi tạo lớp multi-head attention tương ứng với n_heads 
        self.multihead_attn = nn.MultiheadAttention(d_model, n_heads)
        # Khởi tạo Norm layers 
        self.norm1 = nn.LayerNorm(d_model)
        # Khởi tạo feed forward layer
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, 2048),
            nn.ReLU(),
            nn.Linear(2048, d_model),
        )
        # Khởi tạo lớp Norm thứ 2
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Đưa input vào lớp Add & Norm và tính toán qua lớp multi-head attention 
        x = self.norm1(x)
        attn, _ = self.multihead_attn(x, x)
        x = x + attn
        # Tiếp tục đưa x qua lớp Norm và tính toán output của multi head attetion layer vào khối feed forward
        x = self.norm2(x)
        ff = self.feedforward(x)
        x = x + ff
        # Thu được đầu ra của khối  encoder
        return x

