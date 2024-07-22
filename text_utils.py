import re

from typing import List

# Match anything after a "Reference" (or "External Links") header
CHAR_SUB_EXPR = re.compile("[^\x00-\x7F]+")

# Match anything after a "Reference" (or "External Links") header
ENDTEXT_EXPR = re.compile("==+ +?[References|External Links][\\S\\s]*")

# Match common JavaScript and header plaintext tags
WIKITAG_EXPR = re.compile("==+[^=]*?==+|\\n|\\r|{[^}]*?}")

# Match more than one space
EXTRASPACE_EXPR = re.compile("  +")

PUNCT_EXPR = re.compile(r"[^a-z0-9|\.|\?]")

def preprocess_text(text: str, repl: str=" ", stopwords: List[str]=None) -> str:

    # Convert text to lowercase
    text = text.lower()
    # Remove wikipedia-specific tags/unnecessary text
    text = re.sub(ENDTEXT_EXPR, repl, text)
    text = re.sub(WIKITAG_EXPR, repl, text)

    text = re.sub(PUNCT_EXPR, repl, text)

    # this is lazy, but so am i
    text = re.sub(" d ", " had ", text)
    text = re.sub(" s ", " is ", text)
    text = re.sub(" ve ", " have ", text)
    text = re.sub(" m ", " am ", text)
    text = re.sub(" ll ", " will ", text)
    text = re.sub("n t | nt ", " not ", text)

    text = re.sub(EXTRASPACE_EXPR, repl, text)

    if stopwords is not None:
        text = ' '.join([w for w in text.split(' ') if w not in stopwords])

    return text

def load_stopwords(txt_path: str, sep:str=","):
    with open(txt_path, "r") as f:
        text = f.read()
    text = text.strip(" ")
    return list(set(text.split(sep)))
