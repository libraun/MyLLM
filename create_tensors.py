import json
import sys
import os

import argparse

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

from typing import List, Tuple

import text_utils as text

from text_tensor_builder import TextTensorBuilder

import constants

def pretty_print(out: str):

    sys.stdout.write(out)
    sys.stdout.flush()

    sys.stdout.write("\r")
    

# NOTE: maxcount can be any value greater than 0 (does not need to be less than number of messages in dataset)
def get_topicalchat_data(query_config: Chroma | WikipediaQueryRun,
                         json_data: dict,
                         maxcount: int=5000,
                         stopwords: List[str] = None,
                         use_convo_tags: bool = False) -> List[Tuple[Tuple[str, str], str]]:
    
    maxcount_str = str(maxcount)

    conversation_data = list()
    count = 0
    for conversations in json_data.values():

        agent1_data, agent2_data = list(), list()
        for message_data in conversations["content"]:

            msg_text = text.preprocess_text(message_data["message"])
            if message_data["agent"] == "agent_1":
                
                if isinstance(query_config, Chroma):
                    doc = query_config.similarity_search(msg_text)[0].page_content
                else:
                    doc = query_config.run(msg_text)

                doc = text.preprocess_text(doc, stopwords=stopwords)
                agent1_data.append((msg_text, doc))
            else:
                agent2_data.append(msg_text)
            
            count += 1
            pretty_print('/'.join([str(count), maxcount_str]))
            # if target message count has been reached, concatenate past convos with current convos and return.
            if count >= maxcount:
                return conversation_data + list(zip(agent1_data, agent2_data))
            
        convo = list(zip(agent1_data,agent2_data))
        # Optionally concatenate conversation begin and end tags to current conversation.
        # If using these, make sure the data is unshuffled or batched accordingly.
        if use_convo_tags:

            convo[0] = (("<BCONVO> " + convo[0][0][0],convo[0][0][1]), convo[0][1])
            convo[-1] = (convo[-1][0], convo[-1][1] + " <ECONVO>")

        conversation_data += convo
    return conversation_data

def collate(data_batch):

    in_batch, out_batch = zip(*data_batch)
    
    in_batch = pad_sequence(in_batch, padding_value=PAD_IDX)
    out_batch = pad_sequence(out_batch, padding_value=PAD_IDX)
    
    return in_batch,out_batch


parser = argparse.ArgumentParser(
    prog="Create Tensors",
    description="""
    This program queries a Chroma  pytorch dataloaders. It accepts the following arguments:

        --count (-c) INT The max number of messages to process. (Default 10000)
        --batch-size (-b) INT The (unsigned integer) size of each batch for dataloader. (Default 128)
        
        --shuffle (-s) Whether or not to shuffle the dataloaders. (Default False)
        --drop-last-batch (-d) Whether or not to drop the last (incomplete) batch. (Default False)

        --remove-stopwords (-r) Whether or not to remove stopwords from (queried) documents. (Default False)
        --use-wikipedia-api (-w) Whether or not to use the Wikipedia API to directly request documents. (Default False)

        --reverse-encoder-inputs Whether or not to reverse input documents. (Default False)

        --dataset-split-ratio FLOAT A floating point
    """,
    epilog="For more information, see the article on PyTorch DataLoaders")
parser.add_argument("-c", "--count", type=int, default=10000)
parser.add_argument("-b", "--batch-size", type=int, default=128)
parser.add_argument("-s", "--shuffle", default=False,action="store_true")
parser.add_argument("-d", "--drop-last-batch", default=False, action="store_true")
parser.add_argument("-r", "--remove-stopwords", default=False, action="store_true")
parser.add_argument("-w", "--use-wikipedia-api", default=False, action="store_true")
parser.add_argument("--reverse-encoder-inputs", default=False, action="store_true")

parser.add_argument("--dataset-split-ratio", type=float, default=0.8)
# Query chroma using langchain for similar input documents, 
# convert text to tensors and save

if __name__ == "__main__":
    args = parser.parse_args()

    MAX_DATA = args.count
    MODEL_BATCH_SIZE = args.batch_size
    SHUFFLE_DATALOADERS = args.shuffle
    DROP_LAST_BATCH = args.drop_last_batch

    USE_WIKIPEDIA = args.use_wikipedia_api
    REMOVE_STOPWORDS = args.remove_stopwords

    REVERSE_ENCODER_INPUTS = args.reverse_encoder_inputs

    DATASET_SPLIT_RATIO = args.dataset_split_ratio

    assert DATASET_SPLIT_RATIO >= 0.1 and DATASET_SPLIT_RATIO <= 1.0, \
        "For dataset_split_ratio, provide a floating point number greater than 0.1 and <= 1.0"
    
    query_config = None
    # Check if the given (chroma) database path exists and load if so
    if os.path.isdir(constants.DEFAULT_DB_PATH):
        
        embedding_function = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2")

        client = chromadb.PersistentClient(path=constants.DEFAULT_DB_PATH)
        collection = client.get_collection(constants.DEFAULT_DB_NAME)

        query_config = Chroma(
            client=client, 
            collection_name=constants.DEFAULT_DB_NAME,
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
        query_config = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1))
    
    # Stopwords: A list of (very common) strings to remove from the input metadata document.
    stopwords = None if not REMOVE_STOPWORDS else text.load_tokens_from_text("stopwords.txt")

    # Load Topical-Chat dataset from from json
    if not os.path.isfile(constants.TOPICALCHAT_PATH):
        exit(constants.EXIT_FAILURE)

    with open(constants.TOPICALCHAT_PATH, "r") as f:
        json_object = json.load(f)

    # All data is a list containing tuples of the form:
    # ((input_msg, input_doc), output_msg)
    all_data = get_topicalchat_data(
        query_config, json_data=json_object, 
        maxcount=MAX_DATA, stopwords=stopwords) 
    
    vocab_data = list()
    for (in_msg, in_doc), out_msg in all_data:
        vocab_data.append(in_msg)
        vocab_data.append(in_doc)
        vocab_data.append(out_msg)

    # Build and save current vocab for model lookups
    en_vocab = TextTensorBuilder.build_vocab(
        corpus=vocab_data, specials=constants.SPECIAL_TOKENS, 
        save_filepath="en_vocab.pickle")
    
    BOS_IDX, EOS_IDX, PAD_IDX, BMD_IDX, EMD_IDX = tuple(en_vocab.lookup_indices(
        ["<BOS>", "<EOS>", "<PAD>", "<EMD>", "<BMD>"] ))

    # Tensorize docs before creating a DataLoader instance
    tensor_data = list()
    for (agent1_msg, agent1_doc), agent2_msg in all_data:

        in_msg_tensor = torch.cat([
            torch.tensor([BOS_IDX], dtype=torch.long),
            TextTensorBuilder.text_to_tensor(en_vocab, agent1_msg, REVERSE_ENCODER_INPUTS),
            torch.tensor([EOS_IDX], dtype=torch.long)
        ], dim=-1)
        in_doc_tensor = torch.cat([
            torch.tensor([BMD_IDX], dtype=torch.long),
            TextTensorBuilder.text_to_tensor(en_vocab, agent1_doc, REVERSE_ENCODER_INPUTS),
            torch.tensor([EMD_IDX], dtype=torch.long)
        ], dim=-1)

        input_seq = torch.cat([in_msg_tensor, in_doc_tensor], dim=-1)

        output_seq = torch.cat([
            torch.tensor([BOS_IDX], dtype=torch.long),
            TextTensorBuilder.text_to_tensor(en_vocab, agent2_msg),
            torch.tensor([EOS_IDX], dtype=torch.long)
        ], dim=-1)

        tensor_data.append((input_seq, output_seq))

    # Create train/validation splits, save, and exit.
    train_end_idx = math.floor(len(tensor_data) * DATASET_SPLIT_RATIO) 

    train_iter = DataLoader(
        tensor_data[ : train_end_idx], batch_size = MODEL_BATCH_SIZE,
        shuffle = SHUFFLE_DATALOADERS, collate_fn = collate,
        drop_last = DROP_LAST_BATCH)
    valid_iter = DataLoader(
        tensor_data[train_end_idx : ], batch_size = MODEL_BATCH_SIZE,
        shuffle = SHUFFLE_DATALOADERS, collate_fn = collate,
        drop_last = DROP_LAST_BATCH)

    torch.save(train_iter, "./model_tensors/train_tensor_{0}_{1}_{2}.pt".format(
        SHUFFLE_DATALOADERS, MODEL_BATCH_SIZE, MAX_DATA))
    torch.save(valid_iter, "./model_tensors/valid_tensor_{0}_{1}_{2}.pt".format(
        SHUFFLE_DATALOADERS, MODEL_BATCH_SIZE, MAX_DATA))
    
    exit(constants.EXIT_SUCCESS)