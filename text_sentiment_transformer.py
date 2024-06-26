import torch
import torch.nn as nn
import torch.functional as F

class TextSentimentTransformer(nn.Module):

    def __init__(self, input_features: int, 
                 output_features: int, 
                 embed_dim: int, 
                 hidden_dim: int,
                 padding_idx: int):
        
        super().__init__()

        self.embedding = nn.Embedding(
            input_features, embed_dim, 
            padding_idx=padding_idx)

        self.linear_layer = nn.Linear(output_features, hidden_dim)

        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)

        self.fc_out = nn.Linear(hidden_dim, output_features)

    def forward(self, src):

        src = self.embedding(src)
        src = self.linear_layer(src)
        src = self.hidden_layer(src)

        src = self.fc_out(F.relu(src))
        return src
        

        