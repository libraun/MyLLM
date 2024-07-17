import sys
import os
import re

import chromadb

from langchain_chroma import Chroma
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)

import argparse

from constants import DEFAULT_DB_NAME,DEFAULT_DB_PATH, \
                      EXIT_FAILURE, EXIT_SUCCESS, EMBED_DIM



CHAR_SUB_EXPR = re.compile("[^\x00-\x7F]+")



# Main
if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("--preprocess", default=False, type=bool)
    arg_parser.add_argument("--delete", default=False, type=bool)

    arg_parser.parse_args()

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
    
    documents = db_instance.get()["documents"]

    exit(EXIT_SUCCESS)
