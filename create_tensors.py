import json
import sys
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import chromadb

from langchain_chroma import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

#from langchain_community.document_loaders import TextLoader
#from langchain_text_splitters import CharacterTextSplitter
#from model import Encoder, Decoder, Model

import wikipedia_utils

from text_tensor_builder import TextTensorBuilder

from constants import *

def load_data(path: str="./Topical-Chat/conversations/train.json"):
    if not os.path.isfile(path):

        exit(-1)

    with open(path, "r") as f:

        json_object = json.load(f)
    

    conversation_data = list()
    for conversation in json_object.values():

        all_message_data = conversation["content"]

        agent1_data = list()
        agent2_data = list()
        for message_data in all_message_data:

            msg_text = message_data["message"]
            msg_sentiment = message_data["sentiment"]

            msg_text = wikipedia_utils.preprocess_text(msg_text)

            msg_agent = message_data["agent"]
            msg_pair = (msg_text, msg_sentiment)

            if msg_agent == "agent_1":
                agent1_data.append(msg_pair)
            else:
                agent2_data.append(msg_pair)
        conversation_data.append([agent1_data, agent2_data])
    return conversation_data

def collate(data_batch):

    global bos_idx, eos_idx, pad_idx
    global tensor_builder
    in_batch,out_batch = [],[]
    for (in_item, out_item) in data_batch:
        in_batch.append(
            tensor_builder.embedding(
                torch.cat([
                    torch.tensor([bos_idx],dtype=torch.long),
                    in_item,
                    torch.tensor([eos_idx],dtype=torch.long)], dim=0
                )
            )
        )
        out_batch.append(
            tensor_builder.embedding(
                torch.cat([
                    torch.tensor([bos_idx],dtype=torch.long),
                    out_item,
                    torch.tensor([eos_idx],dtype=torch.long)], dim=0
                )
            )
        )
    in_batch = pad_sequence(in_batch, padding_value=pad_idx)
    out_batch = pad_sequence(out_batch, padding_value=pad_idx)

    return in_batch,out_batch

# Query chroma using langchain for similar input documents, 
# convert text to tensors and save

if __name__ == "__main__":

    db_name = DEFAULT_DB_NAME if len(sys.argv) < 2   \
        else sys.argv[1] 
    db_path = DEFAULT_DB_PATH if len(sys.argv) < 3   \
        else sys.argv[2] 
    
    if not os.path.isdir(db_path):
        print("ERROR: DB folder could not be found!")
        exit(EXIT_FAILURE)

    embedding_function = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2")

    client = chromadb.PersistentClient(path="./chroma_data")
    collection = client.get_collection("chatbot")

    db_instance = Chroma(
        client=client, 
        collection_name="chatbot",
        embedding_function=embedding_function
    )
    
    db_documents = db_instance.get()["documents"]
    input_output_data = load_data() 
    
    query_data = list()
    for convo in input_output_data:
        for agent in convo:
            for msg in agent:
                query_data.append(msg[0])
    
    input_documents = db_documents + query_data
                
    tensor_builder = TextTensorBuilder(EMBED_DIM, input_documents)

    tokenizer = tensor_builder.tokenizer
    en_vocab = tensor_builder.get_vocab()

    bos_idx = en_vocab["<bos>"]
    eos_idx = en_vocab["<eos>"]
    pad_idx = en_vocab["<pad>"]

    tensor_data = list()

    data_len_str = str(len(input_output_data))
    count = 0

    input_text_docs, output_text_docs = list(), list()
    for convo in input_output_data[ : 2500]:
        
        agent1_data = convo[0]
        agent2_data = convo[1]

        min_agent_data_length = len(min(agent1_data, agent2_data, key=len))

        for i in range(min_agent_data_length):
            
            input_msg = agent1_data[i][0]
            output_msg = agent2_data[i][0]

            docs = db_instance.similarity_search(input_msg)
            doc = docs[0].page_content

            doc = wikipedia_utils.preprocess_text(doc)

            input_msg_ids = tokenizer(input_msg)
            doc_ids = tokenizer(doc)

            input_doc = input_msg_ids + ["<doc>"] + doc_ids

            agent1_msg_tensor = tensor_builder.convert_text_to_tensor(
                input_msg, tokenize=False)
            agent2_msg_tensor = tensor_builder.convert_text_to_tensor(
                output_msg )
            
            full_doc = (agent1_msg_tensor, agent2_msg_tensor)
            tensor_data.append(full_doc)

            count = count + 1

            sys.stdout.write(str(count) + " / " + data_len_str)
            sys.stdout.flush()

            sys.stdout.write("\r")


    train_data = tensor_data[ : 20000]
    valid_data = tensor_data[ 20000 : 22000 ]
    test_data = tensor_data[ 22000: 23000 ]

    train_iter = DataLoader(train_data, 
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            collate_fn=collate)
    valid_iter = DataLoader(valid_data, 
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            collate_fn=collate)
    test_iter = DataLoader(test_data, 
                           batch_size=BATCH_SIZE,
                           shuffle=True,
                           collate_fn=collate)

    torch.save(train_iter, "./model_tensors/train_tensor.pt")
    torch.save(valid_iter, "./model_tensors/valid_tensor.pt")
    
    torch.save(test_iter, "./model_tensors/test_tensor.pt")

    exit(EXIT_SUCCESS)