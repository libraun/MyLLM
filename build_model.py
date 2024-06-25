import torch
import torch.nn as nn
import torch.optim as optim

from constants import *

def evaluate(iter, model):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for src, trg in iter:
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
    
    train_loss_values = []
    validation_loss_values = []
    for i in range(num_epochs):
        epoch_loss = 0
        model.train() # Set training to true
        for src, trg in train_iter:
            #src, trg = src.to(device), trg.to(device)
            print(src.shape, " ", trg.shape)
            optimizer.zero_grad()
            output = model(src, trg)
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            epoch_loss += loss.item()
        # Add mean loss value as epoch loss.
        epoch_loss = epoch_loss / len(train_iter)
        val_loss = evaluate(valid_iter, model)

        train_loss_values.append(epoch_loss)
        validation_loss_values.append(val_loss)

        if log_msg: # Output epoch progress if log_msg is enabled.
            print(UPDATE_MSG.format(n = i + 1,
                                    t_loss = epoch_loss,
                                    e_loss = val_loss))
    return train_loss_values, validation_loss_values

if __name__ == "__main__":

    train_iter = None
    valid_iter = None

    torch.load(train_iter, "./model_tensors/train_tensor.pt")
    torch.load(valid_iter, "./model_tensors/valid_tensor.pt")
    
    transformer_model = nn.Transformer()

    optimizer = optim.Adam(transformer_model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=1)

    

    train(train_iter, 10, transformer_model, True)