import json
import sys
import os

import math

import chromadb

from typing import List

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from sklearn.feature_extraction.text import TfidfVectorizer

import text_utils
from text_tensor_builder import TextTensorBuilder
from constants import *

def pretty_print_progress(out: str):

    sys.stdout.write(out)
    sys.stdout.flush()

    sys.stdout.write("\r")
    

# NOTE: maxcount can be any value greater than 0 (does not need to be less than number of messages in dataset)
def get_topicalchat_data(
        db_instance, wiki_instance,
        json_data: dict,
        maxcount: int=5000,
        stopwords: List[str] = None):
    
    assert bool(db_instance) != bool(wiki_instance), \
         "ERROR: Supply either a wikipedia instance or a DB instance, not both."

    conversation_data = list()
    count = 0
    for conversations in json_data.values():

        agent1_data, agent2_data = list(), list()
        for i, message_data in enumerate(conversations["content"]):

            msg_text = text_utils.preprocess_text(message_data["message"])
            if message_data["agent"] == "agent_1":
                
                
                if db_instance is not None:
                    doc = db_instance.similarity_search(msg_text)[0].page_content
                else:
                    doc = wiki_instance.run(msg_text)

                doc = text_utils.preprocess_text(doc, stopwords=stopwords)
                agent1_data.append((msg_text, doc))
            else:
                agent2_data.append(msg_text)
            
            count = count + 1
            if count >= maxcount:
                return conversation_data + list(zip(agent1_data, agent2_data))
            pretty_print_progress(str(count) + " / " + str(max_data))

        convo = list(zip(agent1_data,agent2_data))

#        convo[0] = (("<BCONVO_IDX> " + convo[0][0][0],convo[0][0][1]), convo[0][1])
 #       convo[-1] = (convo[-1][0], convo[-1][1] + " <ECONVO_IDX>")
        conversation_data += convo
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
                in_doc_tensor, in_msg_tensor
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

    N_ARGS = len(sys.argv)

    max_data = DEFAULT_MAX_MSGS if N_ARGS < 2 else int(sys.argv[1])

    db_name = DEFAULT_DB_NAME if N_ARGS < 3 else sys.argv[2] 
    db_path = DEFAULT_DB_PATH if N_ARGS < 4 else sys.argv[3] 
    
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
    
    # Stopwords: A list of (very common) strings to remove from the input metadata document.
    
    stopwords = None if not STRIP_INPUT_STOPWORDS else load_stopwords("stopwords.txt")

    # Load Topical-Chat dataset from from json
    if not os.path.isfile(TOPICALCHAT_PATH):
        exit(-1)

    with open(TOPICALCHAT_PATH, "r") as f:
        json_object = json.load(f)

    # All data is a list containing tuples of the form:
    # ((input_msg, input_doc), output_msg)
    all_data = get_topicalchat_data(
        db_instance=db_instance, wiki_instance=wikipedia_instance, 
        json_data=json_object, maxcount=max_data,
        stopwords=stopwords) 

    all_vocab_list = list()
    for (in_msg, in_doc), out_msg in all_data:
        all_vocab_list.append(in_msg)
        all_vocab_list.append(in_doc)
        all_vocab_list.append(out_msg)
    '''
    if BUILD_STOPWORD_LIST:
        
        tfidf_vectorizer = TfidfVectorizer(tokenizer=TextTensorBuilder.tokenizer, stop_words="english")
        tfidf_vec = tfidf_vectorizer.  fit_transform(all_vocab_list)

        stopwords = tfidf_vec.get_stop_words()
        with open("STOPWORDS.txt", "w+") as f:
            f.writelines(stopwords)

        print("Stopwords built!")
        exit(EXIT_SUCCESS)
    '''
    en_vocab = TextTensorBuilder.build_vocab(
        all_vocab_list, SPECIAL_TOKENS, save_filepath="en_vocab.pickle")

    BOS_IDX = en_vocab["<BOS_IDX>"]
    EOS_IDX = en_vocab["<EOS_IDX>"]
    PAD_IDX = en_vocab["<PAD_IDX>"]

    BMD_IDX = en_vocab["<BEGIN_MD_IDX>"]
    EMD_IDX = en_vocab["<END_MD_IDX>"]

    tensor_data = list()
    for (agent1_msg, agent1_doc), agent2_msg in all_data:
        
        # Reverse input documents, according to Seq2Seq paper
        in_msg_tensor = TextTensorBuilder.text_to_tensor(
            en_vocab, agent1_msg, reverse=REVERSE_ENCODER_INPUTS)
        in_md_tensor = TextTensorBuilder.text_to_tensor(
            en_vocab, agent1_doc, reverse=REVERSE_ENCODER_INPUTS)
        
        out_msg_tensor = TextTensorBuilder.text_to_tensor(en_vocab, agent2_msg)
        
        tensor_data.append(((in_msg_tensor, in_md_tensor), out_msg_tensor))

    train_end_idx = math.floor(len(tensor_data) * 0.8) 

    train_data = tensor_data[ : train_end_idx]
    valid_data = tensor_data[ train_end_idx :  ]

    train_iter = DataLoader(train_data, 
                            batch_size=DATALOADER_BATCH_SIZE,
                            shuffle=SHUFFLE_DATALOADERS,
                            collate_fn=collate)
    valid_iter = DataLoader(valid_data, 
                            batch_size=DATALOADER_BATCH_SIZE,
                            shuffle=SHUFFLE_DATALOADERS,
                            collate_fn=collate)

    torch.save(train_iter, "./model_tensors/train_tensor.pt")
    torch.save(valid_iter, "./model_tensors/valid_tensor.pt")

    exit(EXIT_SUCCESS)