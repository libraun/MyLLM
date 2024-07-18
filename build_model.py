import io
import pickle

import torch
import torch.nn as nn
import torch.optim as optim

from constants import *

from torch.nn.utils.rnn import pad_sequence

import model

from constants import *
from create_tensors import collate

def load_tensor(path: str):
    with open(path, "rb") as f:
        buffer = io.BytesIO(f.read())
    return torch.load(buffer)

if __name__ == "__main__":

    train_iter = load_tensor("./model_tensors/train_tensor.pt")
    valid_iter = load_tensor("./model_tensors/valid_tensor.pt")

    with open("vocab.pickle", "rb") as f:
        en_vocab = pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    BOS_IDX = en_vocab["<BOS_IDX>"]
    EOS_IDX = en_vocab["<EOS_IDX>"]
    PAD_IDX = en_vocab["<PAD_IDX>"]

    BMD_IDX = en_vocab["<BEGIN_MD_IDX>"]
    EMD_IDX = en_vocab["<END_MD_IDX>"]

    num_classes = len(en_vocab)

    encoder = model.Encoder(num_classes, HIDDEN_DIM, 
                            padding_idx=PAD_IDX, 
                            device=device).to(device=device)
    decoder = model.Decoder(num_classes, HIDDEN_DIM, 
                            padding_idx=PAD_IDX, 
                            device=device).to(device=device)

    device = torch.device("gpu" if torch.cuda.is_available() else "cpu")

    criterion = nn.CrossEntropyLoss(ignore_index=1)

    model.train_model(encoder, decoder, 
                      PAD_IDX, 
                      train_iter, valid_iter, 
                      criterion, NUM_EPOCHS, 
                      encoder_save_path="encoder.pt",
                      decoder_save_path="decoder.pt",
                      device=device)