import sys
import os

import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from torchtext.data.utils import get_tokenizer

from collections import Counter

import chromadb

from model import Encoder, Decoder, Model

from embedding_builder import EmbeddingBuilder

from constants import DEFAULT_DB_PATH, DEFAULT_DB_NAME, \
                      EXIT_FAILURE, EXIT_SUCCESS, \
                      EMBED_DIM, HIDDEN_DIM

def collate(data_batch):

    global bos_idx, eos_idx, pad_idx
    in_batch,out_batch = [],[]
    for (in_item, out_item) in data_batch:
        in_batch.append(
          torch.cat([
            torch.tensor([bos_idx],dtype=torch.long),
            in_item,
            torch.tensor([eos_idx],dtype=torch.long)],dim=0)
        )
        out_batch.append(torch.cat([
          torch.tensor([bos_idx],dtype=torch.long),
          out_item,
          torch.tensor([eos_idx],dtype=torch.long)],dim=0)
        )
    in_batch = pad_sequence(in_batch, padding_value=pad_idx)
    out_batch = pad_sequence(out_batch, padding_value=pad_idx)

    return in_batch,out_batch

if __name__ == "__main__":

    db_name = DEFAULT_DB_NAME if len(sys.argv) < 2   \
        else sys.argv[1] 
    db_path = DEFAULT_DB_PATH if len(sys.argv) < 3   \
        else sys.argv[2] 
    
    if not os.path.isdir(db_path):
        print("ERROR: DB folder could not be found!")
        exit(EXIT_FAILURE)

    db_retrieve_start = time.time()
    client = chromadb.PersistentClient(path=db_path)
    
    try:
        collection = client.get_collection(db_name)
    except: 
        print("ERROR: Collection couldn't be found!")
        exit(EXIT_FAILURE)

    entries = collection.get()
    
    documents = entries["documents"]
    embeddings = entries["embeddings"]

    emb_builder = EmbeddingBuilder(EMBED_DIM, documents)
'''
    documents = collection.get(include=["documents"])
    documents = [doc for doc in documents["documents"]]
    
    

    embedding_retriever = EmbeddingRetriever(documents)
    en_vocab = embedding_retriever.get_vocab()

    d_model = len(en_vocab)

    encoder = Encoder(d_model,
                      embedding_dim=EMBED_DIM,
                      hidden_dim=HIDDEN_DIM, 
                      padding_idx=en_vocab["<pad>"])
    decoder = Decoder(d_model,
                      embedding_dim=EMBED_DIM, 
                      hidden_dim=HIDDEN_DIM,
                      n_layers=2,
                      padding_idx=en_vocab["<pad>"])
    
    model = Model(encoder, decoder)


    train_iter = DataLoader(train_data, batch_size=BATCH_SIZE,shuffle=True,collate_fn=collate)
    valid_iter = DataLoader(valid_data, batch_size=BATCH_SIZE,shuffle=True,collate_fn=collate)
    test_iter = DataLoader(test_data, batch_size=BATCH_SIZE,shuffle=True,collate_fn=collate)

    exit(EXIT_SUCCESS)
'''