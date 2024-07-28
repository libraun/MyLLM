import random as r
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from constants import MAX_DECODER_OUTPUT_LENGTH, TRAIN_UPDATE_MSG

class Encoder(nn.Module):

    def __init__(self,
                 d_model: int,
                 hidden_dim: int,
                 padding_idx: int,
                 dropout: float=0.5,
                 num_layers: int=2):

        super(Encoder, self).__init__()

        self.msg_rnn = nn.RNN(hidden_dim, hidden_dim,
                              num_layers=num_layers)
        self.md_gru = nn.GRU(hidden_dim, hidden_dim,
                             num_layers=num_layers)

        self.msg_embeddings = nn.Embedding(d_model, hidden_dim,
                                           padding_idx=padding_idx)
        self.doc_embeddings = nn.Embedding(d_model, hidden_dim, 
                                           padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout)

    def forward(self, msg_tensor, md_tensor):

        # Pass apply dropout to document embeddings and pass to gru w/o state
        x1 = self.dropout(self.doc_embeddings(md_tensor))
        _, hidden1 = self.md_gru(x1)

        # Pass message embeddings, and hidden state from md_gru, to msg_rnn
        x2 = self.msg_embeddings(msg_tensor)
        out, hidden2 = self.msg_rnn(x2, hidden1)

        # Take use angle b/w y (hidden2) and x (hidden1)
        hidden = torch.atan2(hidden2, hidden1)
        return out, hidden

class Decoder(nn.Module):

    def __init__(self,
                 output_dim: int,
                 hidden_dim: int,
                 padding_idx: int,
                 device: torch.device,
                 num_layers: int=2,
                 dropout: float = 0.5):

        super(Decoder, self).__init__()

        self.output_dim = output_dim
        self.embeddings = nn.Embedding(output_dim, hidden_dim, 
                                       padding_idx=padding_idx)
        self.dropout = nn.Dropout(0.5)

        self.rnn = nn.GRU(hidden_dim, hidden_dim,
                          num_layers=num_layers,
                          dropout=dropout)

        self.fc_out = nn.Linear(hidden_dim, output_dim)

        self.device = device

    # Teacher-forcing value of 1.0 = always use trg as input; 0.0 = never do that thing
    def forward(self, encoder_outputs, 
                hidden, target_tensor=None,
                teacher_forcing_ratio: float=1.0):

        length = MAX_DECODER_OUTPUT_LENGTH if target_tensor is None \
            else target_tensor.size(-2)

        batch_size = encoder_outputs.size(1)
        decoder_input = torch.zeros(1, batch_size, 
                                    dtype=torch.long, 
                                    device=self.device)
        
        decoder_outputs = torch.zeros(length, batch_size, self.output_dim,
                                      device=self.device)

        for i in range(length):
            decoder_output, hidden = self.forward_step(decoder_input, hidden)
            decoder_outputs[i] = decoder_output
            
            # Use next value of target tensor if teacher_forcing and trg given
            if target_tensor is not None and r.random() > teacher_forcing_ratio:
                decoder_input = target_tensor[i].unsqueeze(0)
            else:
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()
        return decoder_outputs, hidden

    def forward_step(self, input, hidden):

        input = self.dropout(self.embeddings(input))
        input = F.relu(input)

        output, hidden = self.rnn(input, hidden)

        prediction = self.fc_out(output)
        return prediction, hidden


def evaluate_model(encoder: Encoder, 
                   decoder: Decoder, 
                   valid_iter, 
                   criterion: optim.Optimizer,
                   device: torch.device):

    encoder.eval()
    decoder.eval()
    epoch_loss = 0
    with torch.no_grad():
        for src, md, trg in valid_iter:
            src, md, trg = src.to(device), md.to(device), trg.to(device)
            
            encoder_out, hidden = encoder(src, md)
            out, _ = decoder(encoder_out, hidden, trg)

            loss = criterion(out.view(-1, out.size(-1)), trg.view(-1))
            epoch_loss += loss.item()
    return epoch_loss / len(valid_iter)


def train_model(encoder: Encoder,
                decoder: Decoder,
                encoder_optimizer: optim.Optimizer,
                decoder_optimizer: optim.Optimizer,
                train_iter, valid_iter, 
                criterion: optim.Optimizer,
                num_epochs: int,
                encoder_save_path: str,
                decoder_save_path: str,
                device: torch.device,
                log_msg: bool=True):

    train_loss_values = []
    validation_loss_values = []

    for i in range(num_epochs):
        epoch_loss = 0

        decoder.train()
        encoder.train()

        for src, md, trg in train_iter:

            src, md, trg = src.to(device), md.to(device), trg.to(device)

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            encoder_out, hidden = encoder(src, md)

            out, _ = decoder(encoder_out, hidden, trg)

            loss = criterion(out.view(-1, out.size(-1)), trg.view(-1))
            loss.backward()

            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1)

            encoder_optimizer.step()
            decoder_optimizer.step()

            current_loss = loss.item()

            epoch_loss += current_loss
            print(current_loss)
        # Add mean loss value as epoch loss.
        epoch_loss = epoch_loss / len(train_iter)
        val_loss = evaluate_model(encoder, decoder, valid_iter, criterion)

        train_loss_values.append(epoch_loss)
        validation_loss_values.append(val_loss)
        if log_msg: # Output epoch progress if log_msg is enabled.
            print(TRAIN_UPDATE_MSG.format(n = i + 1,
                                    t_loss = epoch_loss,
                                    e_loss = val_loss))
    if encoder_save_path and decoder_save_path:

        torch.save(encoder.state_dict(), encoder_save_path)
        torch.save(decoder.state_dict(), decoder_save_path)
    return train_loss_values, validation_loss_values
