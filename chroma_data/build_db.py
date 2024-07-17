import sys
import os.path
import re
import random
import threading

import chromadb
import wikipedia

from typing import List
from wikipedia.exceptions import DisambiguationError, PageError

from constants import *

import text_utils

CHAR_SUB_EXPR = re.compile("[^\x00-\x7F]+")

def get_data(page_title_list: List[str]) -> None:

    global processed_docs
    
    count = 0
    while count < BATCH_LEN and page_title_list:
        page_title = page_title_list.pop()
        # Attempt to obtain the wiki page referenced by "title"
        try:
            wiki_page = wikipedia.summary(page_title)      
        # On page exception, ignore this page and keep going.
        except (DisambiguationError, PageError) as _:
            continue

        if len(wiki_page) < 5:
            continue

        item = (page_title, wiki_page)

        processed_docs.append(item)
        count = count + 1     

def run_threads(num_threads: int,
                num_pages: int, 
                queries: List[str]) -> None:

    thread_list: List[threading.Thread] = list()
    for _ in range(num_threads):

        query_subset = random.sample(queries, k=num_pages)
        # Set a thread to run "get_data", supplying random_page_titles as arg
        thread = threading.Thread(
            target=get_data,
            args=[query_subset]
        )
        thread_list.append(thread)

    for thread in thread_list:
        thread.start()

    for thread in thread_list:
        thread.join() 

# Main
if __name__ == "__main__":

    db_name = DEFAULT_DB_NAME         \
        if len(sys.argv) < 2          \
        else sys.argv[1] # Name for wiki database
    db_path = DEFAULT_DB_PATH         \
        if len(sys.argv) < 3          \
        else sys.argv[2] # Path to store persistent db

    num_threads = DEFAULT_NUM_THREADS \
        if len(sys.argv) < 4          \
        else int(sys.argv[3])
    num_pages = DEFAULT_NUM_PAGES     \
        if len(sys.argv) < 5          \
        else int(sys.argv[4])

    wikipedia.set_lang("en")

    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(db_name)

    # Load list of topics to query on wikipedia
    topic_list = text_utils.load_topics_from_text("topic_list.txt")

    topic_search_results = text_utils.query_wikipedia_topics(topic_list)
    topic_search_results = [topic for topic in topic_search_results \
                            if "disambiguation" not in topic.lower()]
    topic_search_results = text_utils.filter_topics(collection, topic_search_results)

    random.shuffle(topic_search_results)

    # Retrieve wikipedia pages in rounds
    for _ in range(ROUNDS):

        if LOG_PROGRESS:
            prev_collection_size = collection.count()

        processed_docs = list()
        # Adds wiki page info to global "processed_docs"
        run_threads(num_threads=num_threads, 
                    num_pages=num_pages, 
                    queries=topic_search_results)

        # Pull unique titles and docs from processed documents
        titles = [d[0] for d in processed_docs]
        documents = [d[1] for d in processed_docs]

        titles, documents = text_utils.filter_invalid_pages(titles, documents)

        assert len(titles) == len(documents)

        documents = [text_utils.preprocess_text(doc) for doc in documents]

        collection.add(documents=documents, ids=titles)
        
        if LOG_PROGRESS:
            documents_processed = collection.count() - prev_collection_size
            print("processed", documents_processed, "documents")

    exit(EXIT_SUCCESS)