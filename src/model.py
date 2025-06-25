import torch
import torch.nn as nn
from torchvision import models
import math

class Encoder(nn.Module):
    def __init__(self, embed_dim=512, dropout=0.5):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        # Freeze ResNet parameters
        for param in resnet.parameters():
            param.requires_grad = False
            
        modules = list(resnet.children())[:-2]  # Remove avgpool and fc
        self.resnet = nn.Sequential(*modules)
        
        # Project the feature map to the embedding dimension
        self.projector = nn.Linear(resnet.fc.in_features, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, images):
        # (batch, 3, H, W) -> (batch, 2048, H/32, W/32)
        features = self.resnet(images)
        
        # (batch, 2048, H/32, W/32) -> (batch, H/32 * W/32, 2048)
        batch_size, num_channels, h, w = features.shape
        features = features.permute(0, 2, 3, 1)
        features = features.view(batch_size, h * w, num_channels)
        
        # Project to embedding dimension
        # (batch, H/32 * W/32, 2048) -> (batch, H/32 * W/32, embed_dim)
        features = self.projector(features)
        features = self.relu(features)
        features = self.dropout(features)
        return features

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class DecoderTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, nhead, num_decoder_layers, dim_feedforward, embedding_matrix):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix), freeze=False)
        self.pos_encoder = PositionalEncoding(embedding_dim)
        decoder_layers = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers=num_decoder_layers)
        self.fc_out = nn.Linear(embedding_dim, vocab_size)
        
    def forward(self, captions, features):
        # features: (batch, seq_len_encoder, embed_dim) -> (seq_len_encoder, batch, embed_dim)
        features = features.permute(1, 0, 2)
        
        # captions: (batch, seq_len) -> (seq_len, batch)
        captions = captions.permute(1, 0)

        # embedding and positional encoding
        embeds = self.embedding(captions) * math.sqrt(self.embedding_dim)
        embeds = self.pos_encoder(embeds) # (seq_len, batch, embed_dim)

        # Generate a mask to prevent attention to future tokens
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(len(captions)).to(captions.device)

        # Decode
        output = self.transformer_decoder(embeds, features, tgt_mask=tgt_mask) # (seq_len, batch, embed_dim)
        
        # fc layer
        output = self.fc_out(output) # (seq_len, batch, vocab_size)
        
        # (seq_len, batch, vocab_size) -> (batch, seq_len, vocab_size)
        return output.permute(1, 0, 2)