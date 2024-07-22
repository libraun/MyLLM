import json
import sys
import os

import math

import chromadb

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

import logging

from typing import List

import text_utils as text

from text_tensor_builder import TextTensorBuilder
from constants import *

def pretty_print(out: str):

    sys.stdout.write(out)
    sys.stdout.flush()

    sys.stdout.write("\r")
    

# NOTE: maxcount can be any value greater than 0 (does not need to be less than number of messages in dataset)
def get_topicalchat_data(query_config: Chroma | WikipediaQueryRun,
                         json_data: dict,
                         maxcount: int=5000,
                         stopwords: List[str] = None,
                         use_convo_tags: bool = False):
    
    maxcount_str = str(maxcount)

    conversation_data = list()
    count = 0
    for conversations in json_data.values():

        agent1_data, agent2_data = list(), list()
        for message_data in conversations["content"]:

            msg_text = text.preprocess_text(message_data["message"])
            if message_data["agent"] == "agent_1":
                
                if type(db_instance) is Chroma:
                    doc = db_instance.similarity_search(msg_text)[0].page_content
                else:
                    doc = query_config.run(msg_text)

                doc = text.preprocess_text(doc, stopwords=stopwords)
                agent1_data.append((msg_text, doc))
            else:
                agent2_data.append(msg_text)
            
            count += 1
            # if target message count has been reached, concatenate past convos with current convos and return.
            if count >= maxcount:
                return conversation_data + list(zip(agent1_data, agent2_data))
            else:
                pretty_print('/'.join([str(count), maxcount_str]))
            
        convo = list(zip(agent1_data,agent2_data))
        # Optionally concatenate conversation begin and end tags to current conversation.
        # If using these, make sure the data is unshuffled or batched accordingly.
        if use_convo_tags:

            convo[0] = (("<BCONVO_IDX> " + convo[0][0][0],convo[0][0][1]), convo[0][1])
            convo[-1] = (convo[-1][0], convo[-1][1] + " <ECONVO_IDX>")

        conversation_data += convo
    return conversation_data

def collate(data_batch):

    global BOS_IDX, EOS_IDX, BMD_IDX, EMD_IDX, PAD_IDX
    in_batch, out_batch = *data_batch
    in_batch = pad_sequence(in_batch,padding_value=PAD_IDX)
    out_batch = pad_sequence(out_batch, padding_value=PAD_IDX)
    return in_batch,out_batch

# Query chroma using langchain for similar input documents, 
# convert text to tensors and save

if __name__ == "__main__":

    N_ARGS = len(sys.argv)

    max_data = DEFAULT_MAX_MSGS if N_ARGS < 2 else int(sys.argv[1])

    db_name = DEFAULT_DB_NAME if N_ARGS < 3 else sys.argv[2] 
    db_path = DEFAULT_DB_PATH if N_ARGS < 4 else sys.argv[3] 
    
    db_instance = None

    # Check if the given (chroma) database path exists and load if so
    if os.path.isdir(db_path):
        
        embedding_function = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2")

        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_collection(db_name)

        db_instance = Chroma(
            client=client, 
            collection_name=db_name,
            embedding_function=embedding_function
        )
    # Else, use Wikipedia API to query input data (slow due to requests)
    else:
        logging.warn("""
              WARNING: Persistent ChromaDB instance could not be found.
              Program will use Langchain to directly query Wikipedia from input
              documents, which will be very, very slow.

              In the future, please consider using the sample ChromaDB instance provided,
              or creating your own ChromaDB by adding preprocessed documents to a persistent Chroma instance.
              """)
        db_instance = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    
    # Stopwords: A list of (very common) strings to remove from the input metadata document.
    stopwords = None if not STRIP_INPUT_STOPWORDS else text.load_tokens_from_text("stopwords.txt")

    # Load Topical-Chat dataset from from json
    if not os.path.isfile(TOPICALCHAT_PATH):
        exit(EXIT_FAILURE)

    with open(TOPICALCHAT_PATH, "r") as f:
        json_object = json.load(f)

    # All data is a list containing tuples of the form:
    # ((input_msg, input_doc), output_msg)
    all_data = get_topicalchat_data(
        db_instance=db_instance, json_data=json_object, 
        maxcount=max_data, stopwords=stopwords) 

    # Build and save current vocab for model lookups
    en_vocab = TextTensorBuilder.build_vocab(
        corpus=tuple(sum(all_data)), specials=SPECIAL_TOKENS, 
        save_filepath="en_vocab.pickle")
    
    BOS_IDX, EOS_IDX, PAD_IDX, BMD_IDX, EMD_IDX = en_vocab.lookup_tokens(
        ("<BOS_IDX>", "<EOS_IDX>", "<PAD_IDX>", "<EMD_IDX>", "<BMD_IDX>") )

    # Tensorize docs before creating a DataLoader instance
    tensor_data = list()
    for (agent1_msg, agent1_doc), agent2_msg in all_data:

        # Reverse input documents, according to Seq2Seq paper
        in_msg_tensor = TextTensorBuilder.text_to_tensor(
            en_vocab, agent1_msg, reverse=REVERSE_ENCODER_INPUTS )
        
        in_md_tensor = TextTensorBuilder.text_to_tensor(
            en_vocab, agent1_doc, reverse=REVERSE_ENCODER_INPUTS )
        
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

        input_seq = torch.cat([in_msg_tensor, in_doc_tensor],dim=-1)
        out_msg_tensor = torch.cat([
                torch.tensor([BOS_IDX], dtype=torch.long),
                TextTensorBuilder.text_to_tensor(en_vocab, agent2_msg),
                torch.tensor([EOS_IDX], dtype=torch.long)
        ], dim=-1)

        
        tensor_data.append(((in_msg_tensor, in_md_tensor), out_msg_tensor))

    # Create train/validation splits, save, and exit.
    train_end_idx = math.floor(len(tensor_data) * DATASET_SPLIT_RATIO) 

    train_iter = DataLoader(
        tensor_data[ : train_end_idx], 
        batch_size = DATALOADER_BATCH_SIZE,
        shuffle = SHUFFLE_DATALOADERS, collate_fn=collate)
    valid_iter = DataLoader(
        valid_data = tensor_data[train_end_idx : ], 
        batch_size = DATALOADER_BATCH_SIZE,
        shuffle = SHUFFLE_DATALOADERS, collate_fn=collate)

    torch.save(train_iter, "./model_tensors/train_tensor.pt")
    torch.save(valid_iter, "./model_tensors/valid_tensor.pt")

    exit(EXIT_SUCCESS)