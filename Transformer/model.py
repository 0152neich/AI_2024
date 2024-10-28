from torch import nn
from embedding.positional_encoding import PositionalEncoding
from block.encoder_layer import EncoderLayer
from block.decoder_layer import DecoderLayer

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers):
        super().__init__()
        # Khởi tạo embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        # kHỞi tạo mã hóa vị trí
        self.pos_encoding = PositionalEncoding(d_model)
        # Khởi tạo encoder
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads) for _ in range(n_layers)
        ])
        # Khởi tạo decoder 
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads) for _ in range(n_layers)
        ])
        self.fc = nn.Linear(d_model, vocab_size)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, src, trg):
        # Thực hiện embed input
        src = self.embedding(src)
        # Gắn địa chỉ cho từng giá trị đầu vào
        src = self.pos_encoding(src)

        trg = self.embedding(trg)
        trg = self.pos_encoding(trg)

        # Truyền vào encoder tương ứng với n heads và layers đã khởi tạo
        for layer in self.encoder_layers:
            src = layer(src)

        # Sau khi đưa qua encoder ta tiếp tục truyền đầu ra của encoder vào decoder
        for layer in self.decoder_layers:
            trg = layer(trg, src)

        # Đưa vào lớp Linear
        output = self.fc(trg)
        output = self.softmax(trg)
        return output

