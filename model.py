import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

UPDATE_MSG = "Epoch {n}: Train loss={t_loss:.2f} | Eval loss = {e_loss:.2f}"
MAXLEN = 20

class Encoder(nn.Module):

    def __init__(self,
                 d_model: int,
                # embedding_dim: int,
                 hidden_dim: int,
                # n_layers: int,
                 padding_idx: int,
                 device):

        super(Encoder, self).__init__()

        self.output_layer = nn.GRU(hidden_dim,hidden_dim,
                                   device=device)

        self.embeddings = nn.Embedding(d_model, hidden_dim,
                                       padding_idx=padding_idx,
                                       device=device)
        self.dropout = nn.Dropout(0.5)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):

        x = self.embeddings(x)
        x = self.dropout(x)
        prediction, hidden = self.output_layer(x)

        return prediction, hidden

class Decoder(nn.Module):

    def __init__(self,
                 output_dim: int,
                # embedding_dim: int,
                 hidden_dim: int,
                # n_layers: int,
                 padding_idx: int,
                 device,
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
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

        self.device = device

    def forward(self, encoder_outputs, hidden, target_tensor=None):

        decoder_input = torch.zeros(1,encoder_outputs.size(1),
                                    dtype=torch.long).to(self.device)
        decoder_outputs = []

        length = MAXLEN if target_tensor is None else len(target_tensor)

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

class Model(nn.Module):

    def __init__(self, encoder,decoder, device):

        super(Model, self).__init__()

        self.device = device

        self.encoder = encoder
        self.decoder = decoder

        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

        self.trg_vocab_size = self.decoder.output_dim

    def forward(self, src, trg):

        prediction, hidden = self.encoder.forward(src)
        output, hidden = self.decoder.forward(prediction, hidden, trg)
        return output

    def evaluate_model(self, valid_iter):

        self.eval()
        epoch_loss = 0
        with torch.no_grad():
            for src, trg in valid_iter:
                src,trg = src.to(self.device), trg.to(self.device)
                out = self.forward(src, trg)
                out = out.view(-1, out.shape[-1])

                loss = self.criterion(out, trg.view(-1))
                epoch_loss += loss.item()
        return epoch_loss / len(valid_iter)

    def train_model(self, train_iter, valid_iter, num_epochs: int,
                    model_save_path: str | None = None, log_msg: bool=True):

        train_loss_values = []
        validation_loss_values = []

        for i in range(num_epochs):
            epoch_loss = 0
            self.train()
            for src, trg in train_iter:

                src, trg = src.to(self.device), trg.to(self.device)
                self.encoder.optimizer.zero_grad()
                self.decoder.optimizer.zero_grad()

                out = self.forward(src, trg)
                loss = self.criterion(
                    out.view(-1, out.size(-1)),trg.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
                self.optimizer.step()
                epoch_loss += loss.item()
            # Add mean loss value as epoch loss.
            epoch_loss = epoch_loss / len(train_iter)
            val_loss = self.evaluate_model(valid_iter)

            train_loss_values.append(epoch_loss)
            validation_loss_values.append(val_loss)
            if log_msg: # Output epoch progress if log_msg is enabled.
                print(UPDATE_MSG.format(n = i + 1,
                                        t_loss = epoch_loss,
                                        e_loss = val_loss))
        if model_save_path:
            torch.save(self.state_dict(),model_save_path)
        return train_loss_values, validation_loss_values
