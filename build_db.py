import re
import random
import threading

import chromadb
import wikipedia

from typing import List, Iterable

from langchain_huggingface import HuggingFaceEmbeddings

import text_utils as _text

import itertools

import math

from langchain_chroma import Chroma

CHAR_SUB_EXPR = re.compile("[^\x00-\x7F]+")

processed_docs = list()
m_lock = threading.Lock()

def run_threads(queries: List[str]) -> None:

    thread_list: List[threading.Thread] = list()
    batches = itertools.batched(queries, math.floor(len(queries) / 10))
    for batch in batches:
        # Set a thread to run "get_data", supplying random_page_titles as arg
        thread = threading.Thread(
            target=_get_data,
            args=[batch]
        )
        thread_list.append(thread)
    for thread in thread_list:
        thread.start()
    for thread in thread_list:
        thread.join() 

def _get_data(page_title_list: Iterable[str]) -> None:

    global processed_docs
    global m_lock
    
    local_docs = list()
    for page_title in page_title_list:
        # Attempt to obtain the wiki page referenced by "title"
        try:
            wiki_page = wikipedia.summary(page_title)
        # On page exception (or if a disambiguation page) ignore this page and keep going.
        except wikipedia.exceptions.DisambiguationError | wikipedia.exceptions.PageError:
            continue

        if len(wiki_page) > 5:
            local_docs.append((page_title, wiki_page))
        
    m_lock.acquire()
    processed_docs += local_docs
    m_lock.release()


def _read_lines(path: str) -> List[str]:
    with open(path, "r") as f:
        lines = f.readlines()
    return lines

if __name__ == "__main__":

    # The default names for the database
    db_name: str = "chatbot"
    db_path: str = "chroma_data"

    # Load a list of words to be excluded from input documents from a 
    # (newline-separated) .txt file. 
    stopwords: List[str] = _read_lines("stopwords.txt")
    topics: List[str] = _read_lines("topics.txt")

    # Set default language for searches to english
    wikipedia.set_lang("en")

    embedding_function = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2")

    client = chromadb.PersistentClient(path=db_path)
    collection = Chroma(
        client=client, 
        collection_name=db_name,
        embedding_function=embedding_function
    )

    topic_search_results = list()
    for topic in topics:
        pages = wikipedia.search(topic, results=20)
        topic_search_results += [p for p in pages if not re.search("disambiguation",p.lower())]

    # Remove duplicate wiki page titles, then shuffle
    topic_search_results = list(set(topic_search_results))
    random.shuffle(topic_search_results)

    # Retrieve wikipedia pages in rounds
    for _ in range(10):
        try:
            # Adds wiki page info to global "processed_docs"
            run_threads(queries=topic_search_results)

            # Unzip the list of tuples into two (key and val) columns 
            titles, documents = zip(*processed_docs)
            
            documents = [_text.preprocess(d) for d in documents]
            # Add wiki summaries to chroma collection, using titles as key
            collection.add_texts(texts=documents, ids=[t for t in titles])

            # Clear the global document storage for the next round
            processed_docs.clear() 
        except KeyboardInterrupt:
            break
    exit(0)