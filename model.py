import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):

    def __init__(self, 
                 d_model: int, 
                 embedding_dim: int, 
                 hidden_dim: int, 
                 padding_idx: int):
        
        super(Encoder, self).__init__()

        self.input_layer = nn.Linear(d_model, 
                                     hidden_dim)

        self.output_layer = nn.GRU(hidden_dim, 
                                   hidden_dim)

        self.embeddings = nn.Embedding(d_model, 
                                       embedding_dim,
                                       padding_idx=padding_idx)
        
    def forward(self, x1, x2):

        x = torch.cat( [x1, x2], dtype=torch.long)

        x = self.embeddings(x)
        x, hidden = self.output_layer(x)

        return x, hidden
        
        
class Decoder(nn.Module):
    
    def __init__(self, 
                 output_dim: int, 
                 embedding_dim: int, 
                 hidden_dim: int, 
                 n_layers: int,
                 padding_idx: int, 
                 dropout: float = 0.2):

        super(Decoder, self).__init__()

        self.output_dim = output_dim
        self.embeddings = nn.Embedding(output_dim,
                                       embedding_dim,
                                       padding_idx=padding_idx)

        self.dropout = nn.Dropout(dropout)

        self.input_layer = nn.LSTM(embedding_dim,
                                   hidden_dim,
                                   num_layers=n_layers,
                                   dropout=dropout)
        
        self.output_layer = nn.Linear(embedding_dim,
                                      hidden_dim)
    

    def forward(self, input): 
        
        embs = self.embeddings(input)
        output = self.input_layer(embs)

        prediction = self.output_dim(output)
        return prediction
    
class Model(nn.Module):

    def __init__(self, 
                 encoder: Encoder, 
                 decoder: Decoder, 
                 device=None):

        super(Model, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        
        self.device = device

    def forward(self, src, trg):
        
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
    # FIX
        input_seq, hidden = self.encoder.forward(src)
        for i, input_seq in enumerate(trg):
            output, hidden, cell = self.decoder.forward(input_seq, hidden, cell)
            outputs[i] = output
            
        return outputs