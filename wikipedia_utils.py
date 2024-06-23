import re
import os

import wikipedia
from wikipedia.exceptions import DisambiguationError,PageError

from typing import List

from constants import *

# Match anything after a "Reference" (or "External Links") header
CHAR_SUB_EXPR = re.compile("[^\x00-\x7F]+")

# Match anything after a "Reference" (or "External Links") header
ENDTEXT_EXPR = re.compile("==+ +?[References|External Links][\\S\\s]*")

# Match common JavaScript and header plaintext tags
WIKITAG_EXPR = re.compile("==+[^=]*?==+|\\n|\\r|{[^}]*?}")

# Match more than one space
EXTRASPACE_EXPR = re.compile("  +")

def preprocess_text(text: str, repl: str=" ") -> str:

    # Convert text to lowercase
    text = text.lower()

    # Remove wikipedia-specific tags/unnecessary text
    text = re.sub(ENDTEXT_EXPR, repl, text)
    text = re.sub(WIKITAG_EXPR, repl, text)

    # Remove all extra spaces and non-unicode characters
    text = re.sub(EXTRASPACE_EXPR, repl, text)
    #text = re.sub(BADCHAR_EXPR, repl, text)

    return text

def filter_invalid_pages(titles: List[str], 
                         documents: List[str]):
    i = 0
    while i < len(titles):
        title = titles[i]
        document = documents[i]
        if titles.count(title) != 1 or re.search(CHAR_SUB_EXPR, document):
            
            titles.pop(i)
            documents.pop(i)
        else:
            i = i + 1
    return titles, documents

def load_topics_from_text(path: str, sep="\n") -> None | List[str]:

    if not os.path.exists(path):
        return None

    with open(path, "r") as f:
        text = f.read()

    lines = text.split(sep=sep)
    lines = list(set([line for line in lines if len(line) > 1]))

    return lines

def query_wikipedia_topics(topics: List[str]) -> List[str]:
    
    result = list()
    count = 0
    for topic in topics:
        search_results = wikipedia.search(topic, results=200)

        result += search_results
        count = count + 1

    return result


def filter_topics(db, topic_list: list):

    collection_ids = db.get()["ids"]
    topic_list = [topic for topic in topic_list \
                  if topic not in collection_ids]

    return topic_list

def get_embeddings(document: str,
                   vocab: dict,
                   tokenizer: any,
                   embedding) -> None:
    
    tokens = tokenizer(document)

    result = [vocab[word] for word in tokens]
    result = torch.tensor(result, dtype=torch.long)

    result = embedding(result)
    return result

def build_vocab(corpus: List[str], 
                tokenizer, 
                specials=["<unk>", "<pad>", "<bos>", "<eos>"]):
    counter = Counter()
    for text in corpus:
        tokens = tokenizer(text)
        counter.update(tokens)

    sorted_by_freq_tuples = sorted(counter.items(), 
                                    key=lambda x: x[1], 
                                    reverse=True)
    
    ordered_dict = OrderedDict(sorted_by_freq_tuples)    
    result = vocab(ordered_dict, 
                  specials=specials)
    return result
        
        


    
    







        
     
      
      


