import json
import sys
import os
import pickle
import argparse
import itertools
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
import constants as const

# NOTE: maxcount can be any value greater than 0 (does not need to be less than number of messages in dataset)
def create_data(query_config: Chroma | WikipediaQueryRun,
                conversations: List[List[str]],
                max_similarity_score: float,
                stopwords: List[str] = None,
                maxcount: int = None) -> List[Tuple[Tuple[str, str], str]]:
    

    conversation_data = list()
    for convo in conversations:
        
        if maxcount is not None and len(conversation_data) >= maxcount:
            break

        convo = [text.preprocess(msg) for msg in convo]

        input_output_document = list(itertools.pairwise(convo))

        if len(input_output_document[-1]) != 2:
            input_output_document = input_output_document[ : -1]

        for input_doc, output_doc in input_output_document:
            
            if isinstance(query_config, Chroma):

                md_doc, score = query_config.similarity_search_with_score(input_doc, k=1)

                md_doc = text.preprocess(md_doc[0].page_content, stopwords=stopwords) \
                    if score < max_similarity_score else ""
                
            else:     
                md_doc = text.preprocess(query_config.run(input_doc), stopwords=stopwords)

            conversation_data.append((input_doc, md_doc, output_doc)) 

    return conversation_data


def collate(data_batch):

    in_batch, md_batch, out_batch = zip(*data_batch)
    
    in_batch = pad_sequence(in_batch, padding_value=0)
    md_batch = pad_sequence(md_batch, padding_value=0)
    out_batch = pad_sequence(out_batch, padding_value=0)
    
    return in_batch,md_batch, out_batch


parser = argparse.ArgumentParser(
    prog="Create Tensors",
    description="This program queries a Chroma instance (or WikipediaQueryAPI) to build pytorch dataloaders.",
    usage="""
        The following arguments are accepted.   
        --input-dataset- path (-i)         REQUIRED STR    The filepath to the (.json) training source.
        --input-vocab-path (-v)         STR             Path to an existing torchtext.vocab object to use during DataLoader creation.

        --output-train-path         REQUIRED STR    The filepath to save train DataLoader to (recommended to have a '.pt' extension)
        --output-valid-path         STR             The filepath to save valid DataLoader to (recommended to have a '.pt' extension)

        --output-vocab-path (-v)  STR             Path to store the created vocabulary (do not use in conjunction with "--input-vocab-path")
    
        --count (-c)              REQUIRED INT    The max number of messages to process.
        --batch-size (-b)         REQUIRED INT    The (unsigned integer) size of each batch for dataloader. (Default 64)
        
        --shuffle (-s)            FLAG            Create the dataloaders with the default random batch sampler. (Default False)
        --drop-last-batch (-d)    FLAG            Drop the last (incomplete) batch. (Default False)

        --clean-stopwords         FLAG            Remove stopwords from (queried) documents. (Default False)
        --no-lemmatize            FLAG            Do not lemmatize input words. (Default True)

        --use-wikipedia-api (-w)  FLAG            Use the Wikipedia API to directly request documents. (Default False)

        --reverse-encoder-inputs  FLAG            Reverse input documents (not output documents). (Default False)

        --dataset-split-ratio     FLOAT           How much of the dataset should be reserved for training (between 0.1 and 1). (Default 0.8)

        --max-similarity-score    FLOAT           The maximum similarity score for retrieved documents to be considered 
                (values about max_similarity_score will not be considered). (Default 1.23)
    """,
    epilog="For more information, see the article on PyTorch DataLoaders")

parser.add_argument("-i", "--input-dataset-path", type=str, required=True,
                    help="The path to a (existing) .json dataset.")
parser.add_argument("-v", "--input-vocab-path", type=str, default=None, required=False, 
                    help="Supply the path to a preexisting vocab object to use it during training creation.")

parser.add_argument("--output-train-path", type=str, default="train.pt", required=False, 
                    help="The path (including filename) to store train dataloader.")
parser.add_argument("--output-valid-path", type=str, default="valid.pt", required=False, 
                    help="The path (including filename) to store train dataloader.")
parser.add_argument("--output-vocab-path", type=str, default="en_vocab.sm", required=False, 
                    help="Supply the path to a preexisting vocab object to use it during training creation.")

parser.add_argument("-c", "--count", type=int, required=True,
                    help="Count determines the (lower bound) of messages to be processed.")
parser.add_argument("-b", "--batch-size", type=int, required=True)

parser.add_argument("-s", "--shuffle", default=False, action="store_true", 
                    help="Enabling this feature results in the DataLoader being sampled from randomly every epoch.")
parser.add_argument("-d", "--drop-last-batch", default=False, action="store_true",help="""
                    Enabling this feature results in the last batch of the DataLoader being dropped,
                    potentially avoiding sequences that are just padding.""")
parser.add_argument("-r", "--reverse-encoder-inputs", default=False, action="store_true", 
                    help="Enabling this feature ensures that all encoder inputs are reversed.")

parser.add_argument("--clean-stopwords", default=False, action="store_true", required=False)
parser.add_argument("--no-lemmatize", default=True, action="store_false", required=False)
parser.add_argument("--use-wikipedia-api", default=False, action="store_true", required=False)

parser.add_argument("--dataset-split-ratio", type=float, default=0.8, required=False)
parser.add_argument("--max-similarity-score", type=float, default=1.23, required=False)

# Query chroma using langchain for similar input documents, 
# convert text to tensors and save

if __name__ == "__main__":

    try:
        args = parser.parse_args()
    except:
        exit(const.EXIT_FAILURE)
        
    INPUT_DATASET_PATH: str = args.input_dataset_path
    INPUT_VOCAB_PATH: str = args.input_vocab_path

    OUTPUT_TRAIN_PATH: str = args.train_output_path
    OUTPUT_VALID_PATH: str = args.valid_output_path
    OUTPUT_VOCAB_PATH: str = args.output_vocab_path

    MAX_DATA: int = args.count
    MODEL_BATCH_SIZE: int = args.batch_size
    SHUFFLE_DATALOADERS: bool = args.shuffle
    DROP_LAST_BATCH: bool = args.drop_last_batch

    USE_WIKIPEDIA: bool = args.use_wikipedia_api
    CLEAN_STOPWORDS: bool = args.clean_stopwords

    MAX_SIMILARITY_SCORE: float = args.max_similarity_score
    DATASET_SPLIT_RATIO: float = args.dataset_split_ratio

    REVERSE_ENCODER_INPUTS: bool = args.reverse_encoder_inputs

    assert DATASET_SPLIT_RATIO >= 0.1 and DATASET_SPLIT_RATIO <= 1.0, \
        "For dataset_split_ratio, provide a floating point number greater than 0.1 and <= 1.0"
    
    query_config: Chroma | WikipediaQueryRun = None
    # Check if the given (chroma) database path exists and load if so
    if os.path.isdir(const.DEFAULT_DB_PATH):
        
        embedding_function = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2")

        client = chromadb.PersistentClient(path=const.DEFAULT_DB_PATH)
        collection = client.get_collection(const.DEFAULT_DB_NAME)

        query_config = Chroma(
            client=client, 
            collection_name=const.DEFAULT_DB_NAME,
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
    
    # Stopwords: A list of (usually common) strings to remove from the input metadata document.
    stopwords = None if not CLEAN_STOPWORDS else text.load_tokens_from_text("stopwords.txt")

    # Load Topical-Chat dataset from from json
    if not os.path.isfile(INPUT_DATASET_PATH):
        parser.print_usage()
        exit(const.EXIT_FAILURE)

    with open(INPUT_DATASET_PATH, "r") as f:
        json_object = json.load(f)

    # All data is a list containing tuples of the form:
    # ((input_msg, input_doc), output_msg)
    all_data = create_data(
        query_config, conversations=json_object,  
        max_similarity_score=MAX_SIMILARITY_SCORE,
        stopwords=stopwords, maxcount=MAX_DATA) 
    
    # Create language vocab with flattened all_data
    all_vocab = TextTensorBuilder.build_vocab(
        corpus=itertools.chain.from_iterable(all_data), 
        specials=const.MSG_SPECIAL_TOKENS, 
        default_index_token="<UNK>", min_freq=5,
        save_filepath="all_vocab_sm.pickle")

    # Tensorize docs before creating a DataLoader instance
    tensor_data = list()
    for agent1_msg, agent1_doc, agent2_msg in all_data:

        #
        in_msg_tensor = torch.cat([
            torch.tensor([const.BOS_IDX], dtype=torch.long),

            TextTensorBuilder.text_to_tensor(
                all_vocab, agent1_msg, 
                reverse_tokens=REVERSE_ENCODER_INPUTS),

            torch.tensor([const.EOS_IDX], dtype=torch.long)
        ], dim=-1)
        
        # Wikipedia summaries can get especially long, so we filter out extra tokens
        in_md_tensor = TextTensorBuilder.text_to_tensor(
            all_vocab, agent1_doc, max_tokens=10, reverse_tokens=REVERSE_ENCODER_INPUTS)

        # Cat BOS and EOS tags to out_msg_tensor
        out_msg_tensor = torch.cat([
            torch.tensor([const.BOS_IDX], dtype=torch.long),

            TextTensorBuilder.text_to_tensor(all_vocab, agent2_msg),

            torch.tensor([const.EOS_IDX], dtype=torch.long)
        ], dim=-1)

        tensor_data.append((in_msg_tensor, in_md_tensor, out_msg_tensor))

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

    torch.save(train_iter, TRAIN_OUTPUT_PATH)
    torch.save(valid_iter, VALID_OUTPUT_PATH)
        
    exit(const.EXIT_SUCCESS)