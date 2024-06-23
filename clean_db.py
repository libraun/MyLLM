import sys
import os
import re

import chromadb

from typing import List

from constants import DEFAULT_DB_NAME,DEFAULT_DB_PATH, \
                      EXIT_FAILURE, EXIT_SUCCESS, EMBED_DIM

from embedding_builder import EmbeddingBuilder

import wikipedia_utils


CHAR_SUB_EXPR = re.compile("[^\x00-\x7F]+")

def delete_bad_documents(collection, 
                         batch_size: int = 5000, 
                         max_non_unicode_chars: int = 5) -> None:

    entries = collection.get(include=["documents"])

    collection_docs = entries["documents"]
    collection_ids = entries["ids"]

    num_docs = len(collection_docs)

    total_docs_deleted = 0

    idx = 0
    while idx < num_docs:

        delete_ids_batch = [] # The batch of ids to delete

        batch_end_idx = idx + batch_size # End index for this batch (or num_docs if num_docs is smaller)

        # Append "batch_size" ids to "delete_ids" if doc is dirty.
        while idx < num_docs and idx < batch_end_idx:
            
            current_doc = collection_docs[idx]
            if len(current_doc) < 5:
                delete_ids_batch.append(collection_ids[idx])
                continue

            non_unicode_chars = re.findall(CHAR_SUB_EXPR,current_doc)
            
            if len(non_unicode_chars) > max_non_unicode_chars:
                delete_ids_batch.append(collection_ids[idx])

            idx = idx + 1 # Increment idx

        # If this batch isn't empty, delete this batch and add size of batch to total.
        if delete_ids_batch: 
            collection.delete(ids=delete_ids_batch)
            total_docs_deleted += len(delete_ids_batch)
            
    # Return the number of documents successfully deleted.
    return total_docs_deleted
    

def preprocess_docs(collection, batch_size: int = 500) -> int:

    entries = collection.get(include=["documents"])

    collection_docs = entries["documents"]
    collection_ids = entries["ids"]

    num_docs = len(collection_docs)

    idx = 0 # Index into collection's documents

    # Split docs into batches of size "batch_size" and update 
    # collection using each preprocessed batch of documents 
    while idx < num_docs:

        update_ids_batch, update_docs_batch = list(), list()

        batch_end_idx = idx + batch_size # Index to end this batch on (if smaller than num_docs)
        while idx < batch_end_idx and idx < num_docs:
            
            current_doc = collection_docs[idx]
            current_doc = wikipedia_utils.preprocess_text(current_doc)

            update_ids_batch.append(collection_ids[idx])
            update_docs_batch.append(current_doc)

            idx = idx + 1

        # If this batch is not empty, use it to update collection.
        if update_ids_batch and update_docs_batch:

            collection.update(ids=update_ids_batch, 
                              documents=update_docs_batch)
            print("Cleaned", len(update_docs_batch),"documents.")

def update_embeddings(collection, emb_builder, batch_size: int = 500) -> int:

    entries = collection.get()

    collection_docs = entries["documents"]
    collection_ids = entries["ids"]

    num_docs = len(collection_ids)

    idx = 0 # Index into collection's documents
    # Split docs into batches of size "batch_size" and update 
    # collection using each preprocessed batch of documents 
    while idx < num_docs:

        update_ids_batch = list() 
        update_docs_batch = list()
        update_embeddings_batch = list()

        batch_end_idx = idx + batch_size # Index to end this batch on (if smaller than num_docs)
        while idx < batch_end_idx and idx < num_docs:
            
            current_doc = collection_docs[idx]
            update_ids_batch.append(collection_ids[idx])

            current_embeddings = emb_builder.get_embeddings(current_doc)
            update_embeddings_batch.append(current_embeddings)
            update_docs_batch.append(current_doc)

            idx = idx + 1

        # If this batch is not empty, use it to update collection.
        if update_ids_batch and update_embeddings_batch:

            collection.update(ids=update_ids_batch, 
                              docs=update_docs_batch,
                              embeddings=update_embeddings_batch)
            print("Cleaned", len(update_embeddings_batch),"documents.")

# Main
if __name__ == "__main__":

    db_name = DEFAULT_DB_NAME if len(sys.argv) < 2   \
        else sys.argv[1] 
    db_path = DEFAULT_DB_PATH if len(sys.argv) < 3   \
        else sys.argv[2] 
    
    if not os.path.isdir(db_path):
        print("ERROR: DB folder could not be found!")
        exit(EXIT_FAILURE)

    client = chromadb.PersistentClient(path=db_path)
    
    try:
        collection = client.get_collection(db_name)
    except: 
        print("ERROR: Collection couldn't be found!")
        exit(EXIT_FAILURE)

    documents = collection.get()["documents"]

    emb_builder = EmbeddingBuilder(EMBED_DIM, documents)
    update_embeddings(collection, )
    
    exit(EXIT_SUCCESS)
