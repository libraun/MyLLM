import sys
import os.path
import re
import random
import threading

import chromadb
import wikipedia

from functools import cache
from typing import List
from wikipedia.exceptions import DisambiguationError, PageError

from constants import *

CHAR_SUB_EXPR = re.compile("[^\x00-\x7F]+")

# Match anything after a "Reference" (or "External Links") header
ENDTEXT_EXPR = re.compile("==+ +?[References|External Links][\\S\\s]*")

# Match common JavaScript and header plaintext tags
WIKITAG_EXPR = re.compile("==+[^=]*?==+|\\n|\\r|{[^}]*?}")

# Match more than one space
EXTRASPACE_EXPR = re.compile("  +")

#@cache
def preprocess(text: str, 
               repl: str=" ") -> str:

    # Convert text to lowercase
    text = text.lower()

    # Remove wikipedia-specific tags/unnecessary text
    text = re.sub(ENDTEXT_EXPR, repl, text)
    text = re.sub(WIKITAG_EXPR, repl, text)

    text = text.replace(r"\\",  "")

    # Remove all extra spaces and non-unicode characters
    text = re.sub(EXTRASPACE_EXPR, repl, text)

    return text

def load_strings_from_text(path: str, sep="\n") -> None | List[str]:

    if not os.path.exists(path):
        return None

    with open(path, "r") as f:
        text = f.read()

    lines = text.split(sep=sep)
    lines = [line for line in lines if len(line) > 1]
    return lines


def query_wikipedia_topics(topics: List[str]) -> List[str]:
    
    result = list()
    count = 0
    for topic in topics:
        
        search_results = wikipedia.search(topic, results=200)

        result += search_results
        count = count + 1

    return result


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

        query_set = random.sample(queries, 
                                  k=num_pages)
        # Set a thread to run "get_data", supplying random_page_titles as arg
        thread = threading.Thread(
            target=get_data,
            args=[query_set]
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
    topic_list = load_strings_from_text("topic_list.txt")

    topic_search_results = query_wikipedia_topics(topic_list)
    random.shuffle(topic_search_results)
    
    print(topic_search_results)

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

        for i in range(len(titles)):
            title = titles[i]
            if titles.count(title) > 1:
                titles.pop(i)
                documents.pop(i)

        assert len(titles) == len(documents)

        documents = [preprocess(doc) for doc in documents]

        collection.add(documents=documents, ids=titles)
        
        if LOG_PROGRESS:
            documents_processed = collection.count() - prev_collection_size
            print("processed", documents_processed, "documents")

    exit(EXIT_SUCCESS)