import io
import pickle

import torch
import torch.nn as nn
import torch.optim as optim

import model
from constants import *
from create_tensors import collate

def load_tensor(path: str):
    with open(path, "rb") as f:
        buffer = io.BytesIO(f.read())
    return torch.load(buffer)

if __name__ == "__main__":

    train_iter = load_tensor("./train_ugh_sm.pt")
    valid_iter = load_tensor("./valid_ugh.pt")

    with open("all_vocab_sm.pickle", "rb") as f:
        en_vocab = pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    PAD_IDX = en_vocab["<PAD>"]

    num_classes = len(en_vocab)

    encoder = model.Encoder(num_classes, HIDDEN_DIM, 
                            padding_idx=PAD_IDX).to(device=device)
    decoder = model.Decoder(num_classes, HIDDEN_DIM, 
                            padding_idx=PAD_IDX, 
                            device=device).to(device=device)

    encoder_optimizer = optim.Adam(encoder.parameters())
    decoder_optimizer = optim.Adam(decoder.parameters())
    
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX).to(device)

    model.train_model(encoder, decoder,
                      encoder_optimizer, 
                      decoder_optimizer,
                      train_iter, valid_iter, 
                      criterion, NUM_EPOCHS, 
                      encoder_save_path="encoder_test.pt",
                      decoder_save_path="decoder_test.pt",
                      device=device)