from torch import nn

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        # KHởi tạo lớp multi-head attention tương ứng với n_heads 
        self.multihead_attn1 = nn.MultiheadAttention(d_model, n_heads)
        # Khởi tạo Norm layers 
        self.norm1 = nn.LayerNorm(d_model)
        # KHởi tạo lớp multi-head attention thứ 2 tương ứng với n_heads 
        self.multihead_attn2 = nn.MultiheadAttention(d_model, n_heads)
        # Khởi tạo Norm layers thứ 2
        self.norm2 = nn.LayerNorm(d_model)
        # Khởi tạo feed forward layer
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, 2048),
            nn.ReLU(),
            nn.Linear(2048, d_model),
        )
        # Khởi tạo lớp Norm thứ 3
        self.norm3 = nn.LayerNorm(d_model)
