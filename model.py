import torch
import torch.nn as nn
from torchvision import models

class Encoder(nn.Module):
    def __init__(self, encoded_image_size=256):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]  # Bỏ FC
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(2048, encoded_image_size)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images).squeeze()  # (batch, 2048, 1, 1) -> (batch, 2048)
        features = self.dropout(features)
        features = self.linear(features)  # (batch, 256)
        return features

class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, embedding_matrix):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix), freeze=True)
        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, captions, features):
        embeddings = self.embedding(captions)  # (batch, seq_len, embedding_dim)
        embeddings = self.dropout(embeddings)
        lstm_out, _ = self.lstm(embeddings)  # (batch, seq_len, hidden_dim)
        # Cộng features vào từng bước (broadcast)
        features = features.unsqueeze(1)  # (batch, 1, hidden_dim)
        out = lstm_out + features         # (batch, seq_len, hidden_dim)
        out = self.linear1(out)
        out = self.linear2(out)           # (batch, seq_len, vocab_size)
        return out