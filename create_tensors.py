import json
import sys
import os

import math
import itertools

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

import text_utils

from text_tensor_builder import TextTensorBuilder

from constants import *

import chromadb


def load_topicalchat_data(db_instance, wiki_instance,
              path: str="./Topical-Chat/conversations/train.json",
              maxcount: int=5000):
    if not os.path.isfile(path):

        exit(-1)

    with open(path, "r") as f:

        json_object = json.load(f)
    conversation_data = list()
    for conversation in json_object.values():

        if count >= maxcount:
            break

        all_message_data = conversation["content"]

        agent1_data, agent2_data = list(), list()
        for message_data in all_message_data:

            msg_text = text_utils.preprocess_text(message_data["message"])

            if message_data["agent"] == "agent_1":
                if db_instance is not None:
                    doc = db_instance.similarity_search(msg_text)[0].page_content
                else:
                    doc = wiki_instance.run(msg_text)

                doc = text_utils.preprocess_text(doc)
                agent1_data.append((msg_text, doc))
            else:
                agent2_data.append(msg_text)
            
            if count >= maxcount:
                break
            else:
                count = count + 1
                sys.stdout.write(str(count) + " / " + str(max_data))
                sys.stdout.flush()

                sys.stdout.write("\r")

        conversation_data += itertools.batched(list(zip(agent1_data,agent2_data)),n=2)
        conversation_data.append([agent1_data, agent2_data])
    return conversation_data

def collate(data_batch):

    global BOS_IDX, EOS_IDX, BMD_IDX, EMD_IDX, PAD_IDX
    in_batch,out_batch = [],[]
    for (in_item, out_item) in data_batch:

        in_msg_tensor, in_doc_tensor = in_item
        in_msg_tensor = torch.cat([
            torch.tensor([BOS_IDX], dtype=torch.long),
            in_msg_tensor,
            torch.tensor([EOS_IDX], dtype=torch.long)
        ], dim=-1)
        in_doc_tensor = torch.cat([
            torch.tensor([BMD_IDX], dtype=torch.long),
            in_doc_tensor,
            torch.tensor([EMD_IDX], dtype=torch.long)
        ], dim=-1)

        in_batch.append(
            torch.cat([
                in_msg_tensor, in_doc_tensor
            ], dim=-1)
        )
        out_batch.append(
            torch.cat([
                torch.tensor([BOS_IDX], dtype=torch.long),
                out_item,
                torch.tensor([EOS_IDX], dtype=torch.long)
            ], dim=-1)
        )
    in_batch = pad_sequence(in_batch,padding_value=PAD_IDX)
    out_batch = pad_sequence(out_batch, padding_value=PAD_IDX)
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
    
    wikipedia_instance = None
    if os.path.isdir(db_path):

        embedding_function = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2")

        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_collection(db_name)

        db_instance = Chroma(
            client=client, 
            collection_name="chatbot",
            embedding_function=embedding_function
        )
    else:
        print("""
              WARNING: Persistent ChromaDB instance could not be found.
              Program will use Langchain wikipedia to directly query input documents, 
              which will be very, very slow.

              In the future, please consider adding preprocessed documents
              to a persistent chromaDB instance, and loading the sentence embeddings from there.
              """)
        wikipedia_instance = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        
    
    all_data,agent1_msgs, agent2_msgs = load_data(
        db_instance=db_instance, wiki_instance=wikipedia_instance) 

    stopwords = load_stopwords("stopwords.txt")

    agent1_special_tokens = [
        "<UNK_IDX>", "<PAD_IDX>",
        "<BOS_IDX>", "<EOS_IDX>", 
        "<BEGIN_MD_IDX>", "<END_MD_IDX>"
    ]

    all_vocab_list = list()
    for (agent1_msg, agent1_md_doc), agent2_msg in all_data:
        
        all_vocab_list.append(agent1_msg)
        all_vocab_list.append(agent1_md_doc)

        all_vocab_list.append(agent2_msg)

    en_vocab = TextTensorBuilder.build_vocab(all_vocab_list, specials=agent1_special_tokens)

    TextTensorBuilder.save_vocab(en_vocab, "all_vocab.pickle")

    BOS_IDX = en_vocab["<BOS_IDX>"]
    EOS_IDX = en_vocab["<EOS_IDX>"]
    PAD_IDX = en_vocab["<PAD_IDX>"]

    BMD_IDX = en_vocab["<BEGIN_MD_IDX>"]
    EMD_IDX = en_vocab["<END_MD_IDX>"]

    tensor_data = list()

    wikipedia_instance = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    for (agent1_msg, agent1_doc), agent2_msg in all_data:
        
        in_msg_tensor = TextTensorBuilder.text_to_tensor(agent1_msg)
        in_md_tensor = TextTensorBuilder.text_to_tensor(agent1_doc)
        
        out_msg_tensor = TextTensorBuilder.text_to_tensor(agent2_msg)
        
        tensor_data.append(((in_msg_tensor, in_md_tensor), out_msg_tensor))


    train_end_idx = math.floor(len(tensor_data) * 0.8) 
    valid_end_idx = math.floor(len(tensor_data) * 0.9)

    train_data = tensor_data[ : train_end_idx]
    valid_data = tensor_data[ train_end_idx : valid_end_idx ]
    test_data = tensor_data[ valid_end_idx : ]

    train_iter = DataLoader(train_data, 
                            batch_size=64,
                            shuffle=False,
                            collate_fn=collate)
    valid_iter = DataLoader(valid_data, 
                            batch_size=64,
                            shuffle=False,
                            collate_fn=collate)
    test_iter = DataLoader(test_data, 
                           batch_size=64,
                           shuffle=False,
                           collate_fn=collate)

    torch.save(train_iter, "./model_tensors/train_tensor.pt")
    torch.save(valid_iter, "./model_tensors/valid_tensor.pt")
    torch.save(test_iter, "./model_tensors/test_tensor.pt")

    exit(EXIT_SUCCESS)