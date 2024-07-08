import io
import sys
import pickle

import torch
import torch.nn as nn
import torch.optim as optim

from constants import *


from torch.nn.utils.rnn import pad_sequence

import model


def evaluate():
    global seq2seq, valid_iter, criterion
    seq2seq.eval()
    epoch_loss = 0
    with torch.no_grad():
        for src, trg in valid_iter:
          #  src,trg = src.to(device), trg.to(device)
            output = model(src, trg)
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(iter)

UPDATE_MSG = "Epoch {n}: Train loss={t_loss:.2f} | Eval loss = {e_loss:.2f}"
def train(train_iter, 
          num_epochs: int, 
          model,
          log_msg=True):
    global seq2seq, optimizer, criterion

    train_loss_values = []
    validation_loss_values = []
    for _ in range(num_epochs):
        epoch_loss = 0
        seq2seq.train() # Set training to true
        for j, (src, trg) in enumerate(train_iter):
      #      src, trg = src.to(device), trg.to(device)
            
            optimizer.zero_grad()
            output = model(src, trg)
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(seq2seq.parameters(), 1)
            optimizer.step()
            epoch_loss += loss.item()
            if log_msg: # Output epoch progress if log_msg is enabled.
                sys.stdout.write(str(j) + " / " + str(len(train_iter)))
                sys.stdout.flush()

                sys.stdout.write("\r")
        # Add mean loss value as epoch loss.
        epoch_loss = epoch_loss / len(train_iter)
        val_loss = evaluate()

        train_loss_values.append(epoch_loss)
        validation_loss_values.append(val_loss)
    return train_loss_values, validation_loss_values

def load_tensor(path: str):
    with open(path, "rb") as f:
        buffer = io.BytesIO(f.read())
    return torch.load(buffer)

def collate(data_batch):

    bos_idx = 2
    eos_idx = 3
    pad_idx = 1
    in_batch,out_batch = [],[]
    for (in_item, out_item) in data_batch:
        in_batch.append(
            torch.cat([
                torch.tensor([bos_idx],dtype=torch.long),
                in_item,
                torch.tensor([eos_idx],dtype=torch.long)], dim=0
            )
            
        )
        out_batch.append(
            torch.cat([
                torch.tensor([bos_idx],dtype=torch.long),
                out_item,
                torch.tensor([eos_idx],dtype=torch.long)], dim=0
            )
        )
    in_batch = pad_sequence(in_batch, padding_value=pad_idx)
    out_batch = pad_sequence(out_batch, padding_value=pad_idx)

    return in_batch,out_batch

if __name__ == "__main__":

    train_iter = load_tensor("./model_tensors/train_tensor.pt")
    valid_iter = load_tensor("./model_tensors/valid_tensor.pt")

    with open("vocab.pickle", "rb") as f:
        en_vocab = pickle.load(f)

 #   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    encoder = model.Encoder(len(en_vocab), 8, 4, 2, 1)#.to(device=device)
    decoder = model.Decoder(len(en_vocab), 8, 4, 2, 1)#.to(device=device)

    device = torch.device("gpu" if torch.cuda.is_available() else "cpu")

    seq2seq = model.Model(encoder, decoder)
    print('yo')

    optimizer = optim.Adam(seq2seq.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=1)

    train(train_iter, 10, seq2seq, True)