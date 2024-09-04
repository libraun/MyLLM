import pickle
import yaml

import os
import itertools
import math

import torch
from torch.utils.data import DataLoader

from torch.nn.utils.rnn import pad_sequence

import text_utils as text
import constants as const
from text_tensor_builder import TextTensorBuilder

from ChromaInstance import ChromaInstance


def collate(data_batch):

    in_batch, out_batch = [], []
    for in_msg, out_msg in data_batch:

        in_batch.append(in_msg)
        out_batch.append(out_msg)
        
    in_batch = pad_sequence(in_batch, padding_value=const.PAD_IDX)
    out_batch = pad_sequence(out_batch, padding_value=const.PAD_IDX)
    
    return in_batch, out_batch


# Query chroma using langchain for similar input documents, 
# convert text to tensors and save

if __name__ == "__main__":
    
    # Get variable parameters from "config.yaml"
    with open("config.yaml", "rb") as f:
        app_config = yaml.safe_load(f)["create_tensors"]

    # Raise an error if the dataset_split_ratio in app_config is invalid.
    if app_config["dataset_split_ratio"] <= 0.1 or app_config["dataset_split_ratio"] >= 1.0:
        raise "For dataset_split_ratio, provide a floating point number greater than 0.1 and <= 1.0"
    
    # "query_config" is the configuration used to gather documents for augmentation.
    # It can either be a ChromaDB instance, or a WikipediaQueryRun instance (will query Wikipedia directly(slow))

        
    query_config = ChromaInstance(app_config["db_path"], app_config["db_name"])
    # Use Wikipedia API to query input data (slow due to requests) INSTEAD of LangChain

    # Stopwords: A list of (usually common) strings to remove from the input metadata document.
    if app_config["stopwords_txt_path"] is not None:
        stopwords = text.load_tokens_from_text(app_config["stopwords_txt_path"])
    else:
        stopwords = None

    # Raise an error if given dataset path DNE
    if not os.path.isfile(app_config["input_dataset_path"]):
        raise "Input dataset path not found; quitting."

    # Open the input dataset file and unpickle it.
    with open(app_config["input_dataset_path"], "rb") as f:
        # First "flatten" lists
        data = list(itertools.chain(pickle.load(f)))

    # Sort data into pairs
    data = [(data[i], data[i+1]) for i in range(0, len(data)-2, 2)]

    all_data = list()
    # Iterate through statement/response pairs to build RAG data
    for input_doc, output_doc in data:

        md_doc = query_config.query_similar_document(
            input_doc,app_config["max_similarity_score"])
        
        all_data.append((input_doc, md_doc, output_doc))
    
    # Create Vocab object using list of strings, saving result to 
    # the "output_vocab_path" specified in config.yaml
    all_vocab = TextTensorBuilder.build_vocab(
        corpus=itertools.chain.from_iterable(all_data), 
        specials=const.MSG_SPECIAL_TOKENS, 
        default_index_token="<UNK>", min_freq=5,
        save_filepath=app_config["output_vocab_path"])

    # Convert strings to tensors to pass into dataloader instance.
    tensor_data = list()
    for agent1_msg, agent1_doc, agent2_msg in all_data:

        # Concatenate beginning-of-sentence (BOS) and end-of-sentence (EOS) indices to the input query.
        in_msg_tensor = torch.cat([
            torch.tensor([const.BOS_IDX], dtype=torch.long),
            TextTensorBuilder.text_to_tensor(
                all_vocab, agent1_msg, 
                reverse_tokens=app_config["reverse_encoder_inputs"],
                max_tokens=20),
            torch.tensor([const.EOS_IDX], dtype=torch.long)
        ], dim=-1)
        
        # Check if the web-retrieved document does not consist of a single "<PAD>" token
        # before concattenating it to the input query.
        if len(agent1_doc.split()) > 1:

            in_md_tensor = TextTensorBuilder.text_to_tensor(
                all_vocab, agent1_doc, max_tokens=20, 
                reverse_tokens=app_config["reverse_encoder_inputs"])
            in_msg_tensor = torch.cat([in_msg_tensor, in_md_tensor], dim=-1)

        # Concatenate BOS and EOS tags to target response (out_msg_tensor)
        out_msg_tensor = torch.cat([
            torch.tensor([const.BOS_IDX], dtype=torch.long),

            TextTensorBuilder.text_to_tensor(all_vocab, agent2_msg, max_tokens=20),

            torch.tensor([const.EOS_IDX], dtype=torch.long)
        ], dim=-1)  

        # Finally, append a tuple containing the input user message and target response
        # to the list of tensors.
        tensor_data.append((in_msg_tensor, out_msg_tensor))

    # Create train/validation splits, save, and exit.
    train_end_idx = math.floor(len(tensor_data) * app_config["dataset_split_ratio"]) 

    # Create training DataLoader and valid DataLoader, respectively.
    train_iter = DataLoader(
        tensor_data[ : train_end_idx], batch_size = app_config["batch_size"],
        shuffle = app_config["shuffle_dataloaders"], collate_fn = collate,
        drop_last = app_config["drop_last_batch"],)
    valid_iter = DataLoader(
        tensor_data[train_end_idx : ], batch_size = app_config["batch_size"],
        shuffle = app_config["shuffle_dataloaders"], collate_fn = collate,
        drop_last = app_config["drop_last_batch"],)

    # Save both dataloaders to path specified by "config.yaml"
    torch.save(train_iter, app_config["output_train_path"])
    torch.save(valid_iter, app_config["output_valid_path"])
    
    # Exit successfully
    exit(const.EXIT_SUCCESS)