import torch
import torch.nn as nn

class Encoder(nn.Module):

    def __init__(self, 
                 d_model: int, 
                 embedding_dim: int, 
                 hidden_dim: int, 
                 padding_idx: int,
                 n_layers: int):
        
        super(Encoder, self).__init__()

        self.input_layer = nn.Linear(d_model, 
                                     hidden_dim)

        self.output_layer = nn.LSTM(embedding_dim, 
                                    hidden_dim,
                                    num_layers=n_layers)

        self.embeddings = nn.Embedding(d_model, 
                                       embedding_dim,
                                       padding_idx=padding_idx)
        
    def forward(self, x):

        x = self.embeddings(x)
        _, (hidden, cell) = self.output_layer(x)

        return hidden, cell
        
        
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

        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           dropout=dropout)
        
        self.fc_out = nn.Linear(hidden_dim,
                                output_dim)
    

    def forward(self, input, hidden, cell): 

        input = self.embeddings(input.unsqueeze(0))
        input = self.dropout(input)

        output, (hidden, cell) = self.input_layer(input)

        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, cell
    
class Model(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder):

        super(Model, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.trg_vocab_size = self.decoder.output_dim

    def forward(self, src, trg):
        
        trg_len, batch_size = trg.shape

        outputs = torch.zeros(trg_len, batch_size, self.trg_vocab_size)
        hidden, cell = self.encoder.forward(src)
        for i, input_seq in enumerate(trg):
            output, hidden, cell = self.decoder.forward(input_seq, hidden, cell)
            outputs[i] = output
        return outputs
    

    def predict(self, src, 
                shape: tuple,
                eos_idx: int=3, 
                maxlength: int=10):

        trg_len, batch_size, vocab_size = shape
        with torch.no_grad():
            src = src.to(self.device)
            result = list()
            for _ in range(maxlength):

                trg_pred = torch.zeros(
                    trg_len, batch_size, vocab_size
                ).to(self.device)

                output = self.forward(src,trg_pred)

                output = output[1:].argmax(1)

                output = output.view(trg_len, batch_size)

                top1 = output[1:].argmax(-1).tolist()[0]
                result.append(top1)
        return result