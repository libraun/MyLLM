import sys
import re
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

@cache
def preprocess(text: str, repl: str=" ") -> str:

    # Convert text to lowercase
    text = text.lower()

    # Remove wikipedia-specific tags/unnecessary text
    text = re.sub(ENDTEXT_EXPR, repl, text)
    text = re.sub(WIKITAG_EXPR, repl, text)

    # Remove all extra spaces and non-unicode characters
    text = re.sub(EXTRASPACE_EXPR, repl, text)

    return text

def get_data(page_title_list: List[str]):

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
        
        if re.search(CHAR_SUB_EXPR, wiki_page):
            continue
        
        summary = preprocess(wiki_page)

        item = (page_title, summary)

        processed_docs.append(item)
        count = count + 1     


def run_threads(num_threads: int,
                num_pages: int):

    thread_list: List[threading.Thread] = list()
    for _ in range(num_threads):

        # Generate random page titles for this threadess
        random_page_titles = wikipedia.random(num_pages)
        # Set a thread to run "get_data", supplying random_page_titles as arg
        thread = threading.Thread(
            target=get_data,
            args=[random_page_titles]
        )
        thread.start()
        thread_list.append(thread)
        
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

    # Retrieve wikipedia pages in rounds
    for _ in range(ROUNDS):

        if LOG_PROGRESS:
            prev_collection_size = collection.count()

        processed_docs = list()
        # Adds wiki page info to global "processed_docs"
        run_threads(num_threads, num_pages)

        # Pull unique titles and docs from processed documents
        titles = list(set([d[0] for d in processed_docs]))
        documents = [d[1] for d in processed_docs]

        collection.add(documents=documents, ids=titles)
        
        if LOG_PROGRESS:
            documents_processed = collection.count() - prev_collection_size
            print("processed", documents_processed, "documents")

    exit(EXIT_SUCCESS)