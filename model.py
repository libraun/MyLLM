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
                 device: torch.device,
                 dropout: float=0.5):

        super(Encoder, self).__init__()

        self.output_layer = nn.GRU(hidden_dim,hidden_dim,
                                   device=device)

        self.embeddings = nn.Embedding(d_model, hidden_dim,
                                       padding_idx=padding_idx,
                                       device=device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        x = self.dropout(self.embeddings(x))
        prediction, hidden = self.output_layer(x)

        return prediction, hidden

class Decoder(nn.Module):

    def __init__(self,
                 output_dim: int,
                 hidden_dim: int,
                 padding_idx: int,
                 device: torch.device,
                 dropout: float = 0.5):

        super(Decoder, self).__init__()

        self.output_dim = output_dim
        self.embeddings = nn.Embedding(
            output_dim, hidden_dim, padding_idx=padding_idx,
            device=device)

        self.dropout = nn.Dropout(dropout)

        self.rnn = nn.GRU(hidden_dim,hidden_dim,
                          device=device)

        self.fc_out = nn.Linear(hidden_dim, output_dim,
                                device=device)

        self.device = device

    def forward(self, encoder_outputs, hidden, target_tensor=None):

        length = MAX_DECODER_OUTPUT_LENGTH if target_tensor is None else len(target_tensor)

        batch_size = encoder_outputs.size(1)
        decoder_input = torch.ones(1, batch_size, dtype=torch.long).to(self.device)
        
        decoder_outputs = []

        for i in range(length):
            decoder_output, hidden = self.forward_step(decoder_input, hidden)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                decoder_input = target_tensor[i].unsqueeze(0)
            else:
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()
        decoder_outputs = torch.cat(decoder_outputs, dim=0)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, hidden

    def forward_step(self, input, hidden):

        input = self.embeddings(input)
        input = F.relu(self.dropout(input))

        output, hidden = self.rnn(input, hidden)

        prediction = self.fc_out(output)
        return prediction, hidden


def evaluate_model(encoder: Encoder, 
                   decoder: Decoder, 
                   valid_iter, 
                   criterion,
                   device: torch.device):

    encoder.eval()
    decoder.eval()
    epoch_loss = 0
    with torch.no_grad():
        for src, trg in valid_iter:
            src,trg = src.to(device), trg.to(device)
            prediction, hidden = encoder(src)
            out, _ = decoder(prediction, hidden, trg)

            loss = criterion(out.view(-1, out.size(-1)), trg.view(-1))
            epoch_loss += loss.item()
    return epoch_loss / len(valid_iter)


def train_model(encoder: Encoder,
                decoder: Decoder,
                train_iter, 
                valid_iter, 
                criterion,
                num_epochs: int,
                encoder_save_path: str,
                decoder_save_path: str,
                device: torch.device,
                log_msg: bool=True):

    train_loss_values = []
    validation_loss_values = []

    encoder_optimizer = optim.Adam(encoder.parameters())
    decoder_optimizer = optim.Adam(decoder.parameters())

    for i in range(num_epochs):
        epoch_loss = 0

        decoder.train()
        encoder.train()

        for src, trg in train_iter:

            src, trg = src.to(device), trg.to(device)
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            prediction, hidden = encoder(src)

            out, _ = decoder(prediction, hidden, trg)

            loss = criterion(out.view(-1, out.size(-1)), trg.view(-1))
            loss.backward()

            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1)

            encoder_optimizer.step()
            decoder_optimizer.step()
            epoch_loss += loss.item()
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
