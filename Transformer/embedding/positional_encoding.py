import torch
from torch import nn

class PositionalEncoding(nn.Module):
    """
        Tính toán hàm sinusoid
    """

    def __init__(self, d_model: int, max_len: int=5000) -> None:
        """
            d_model: chiều của model
            max_len: chiều dài tối đa sequence
        """

        super(PositionalEncoding, self).__init__()

        self.encoding = torch.zeros(max_len, d_model)

        pos = torch.arange(0, max_len)
        pos = pos.float().unsqueeze(dim=1)

        _2i = torch.arange(0, d_model, step=2).float()

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):

        batch_size, seq_len = x.size()
        x = x + self.encoding[:, :seq_len]
