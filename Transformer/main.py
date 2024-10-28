from torchinfo import summary
from model import Transformer
# Khởi tạo mô hình
model = Transformer(vocab_size=10000, d_model=512, n_heads=8, n_layers=6)

# Hiển thị kiến trúc của mô hình
summary(model) # batch size là 32, sequence length là 100