import json
import sys
import os

import math

import pickle

import torch
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

def load_data(use_langchain_query: bool=False,
              path: str="./Topical-Chat/conversations/train.json"):
    if not os.path.isfile(path):

        exit(-1)

    with open(path, "r") as f:

        json_object = json.load(f)

    conversation_data = list()
    for conversation in json_object.values():

        all_message_data = conversation["content"]

        agent1_data, agent2_data = list(), list()
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

def load_stopwords(txt_path: str, sep:str=","):
    with open(txt_path, "r") as f:
        text = f.read()
    text = text.strip(" ")
    return text.split(sep) 

# Query chroma using langchain for similar input documents, 
# convert text to tensors and save

if __name__ == "__main__":

    max_data = int(sys.argv[1])

    db_name = DEFAULT_DB_NAME if len(sys.argv) < 3   \
        else sys.argv[2] 
    db_path = DEFAULT_DB_PATH if len(sys.argv) < 4   \
        else sys.argv[3] 
    
    use_langchain_query = False
    if os.path.isdir(db_path):

        embedding_function = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2")

        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_collection(db_name)

        db_instance = Chroma(
            client=client, 
            collection_name="chatbot",
            embedding_function=embedding_function
        )

        db_documents = db_instance.get()["documents"]
    else:
        print("""
              WARNING: Persistent ChromaDB instance could not be found.
              Program will use Langchain wikipedia to directly query input documents, 
              which will be very, very slow.

              In the future, please consider adding preprocessed documents
              to a persistent chromaDB instance, and loading the sentence embeddings from there.
              """)
        use_langchain_query = True
        
    
    input_output_data = load_data(use_langchain_query=use_langchain_query) 

    stopwords = load_stopwords("stopwords.txt")

    query_data = list()
    for convo in input_output_data:
        for agent in convo:
            for msg in agent:
                query_data.append(msg[0])
    
    input_documents = db_documents + query_data
                
    tensor_builder = TextTensorBuilder(input_documents, init_save_path="vocab.pickle", stopwords=stopwords)

    tokenizer = tensor_builder.tokenizer
    en_vocab = tensor_builder.lang_vocab

    bos_idx = en_vocab["<BOS_IDX>"]
    eos_idx = en_vocab["<EOS_IDX>"]
    pad_idx = en_vocab["<PAD_IDX>"]

    tensor_data = list()
    count = 0
    for agent1_data, agent2_data in input_output_data:

        if count >= max_data:
            break

        min_agent_data_length = len(min(agent1_data, agent2_data, key=len))

        for i in range(min_agent_data_length):
            
            input_msg = agent1_data[i][0]
            output_msg = agent2_data[i][0]

            docs = db_instance.similarity_search(input_msg)
            doc = docs[0].page_content

            doc = wikipedia_utils.preprocess_text(doc)

            input_msg_ids = [w for w in tokenizer(input_msg) if w not in stopwords]
            doc_ids = [w for w in tokenizer(doc) if w not in stopwords]

            input_doc = input_msg_ids + ["<BEGIN_MD>"] + doc_ids + ["<END_MD>"]

            print(input_doc)

            agent1_msg_tensor = tensor_builder.convert_text_to_tensor(
                input_msg, tokenize=False)
            agent2_msg_tensor = tensor_builder.convert_text_to_tensor(
                output_msg )
            
            full_doc = (agent1_msg_tensor, agent2_msg_tensor)
            tensor_data.append(full_doc)

            count = count + 1

            if count >= max_data:
                break

            sys.stdout.write(str(count) + " / " + str(max_data))
            sys.stdout.flush()

            sys.stdout.write("\r")

    train_end_idx = math.floor(len(tensor_data) * 0.8) 
    valid_end_idx = math.floor(len(tensor_data) * 0.9)

    train_data = tensor_data[ : train_end_idx]
    valid_data = tensor_data[ train_end_idx : valid_end_idx ]
    test_data = tensor_data[ valid_end_idx : ]

    train_iter = DataLoader(train_data, 
                            batch_size=32,
                            shuffle=False,
                            collate_fn=collate)
    valid_iter = DataLoader(valid_data, 
                            batch_size=32,
                            shuffle=False,
                            collate_fn=collate)
    test_iter = DataLoader(test_data, 
                           batch_size=32,
                           shuffle=False,
                           collate_fn=collate)

    torch.save(train_iter, "./model_tensors/train_tensor.pt")
    torch.save(valid_iter, "./model_tensors/valid_tensor.pt")
    torch.save(test_iter, "./model_tensors/test_tensor.pt")

    exit(EXIT_SUCCESS)