import re
import nltk
from torchtext.data.utils import get_tokenizer

from typing import List

# Match more than one space
EXTRASPACE_EXPR = re.compile("  +")

# Match all non-alphanumeric chars EXCEPT FOR period ('.') and question mark (?)
PUNCT_EXPR = re.compile(r"[^a-z0-9|\.|\?]")

TRANSLATE_TABLE = { " d " : " had ", " s " : " is ", " ve " : " have ",
                    " m " : " am ", " ll ": " will ", "n t ": " not " , " nt " : " not "}
NLTK_LEMMATIZER = nltk.WordNetLemmatizer()


tokenizer = get_tokenizer("spacy","en_core_web_sm")
def preprocess(text: str, repl: str=" ",
               stopwords: List[str]=None, 
               lemmatize: bool=True) -> str:

    # Convert text to lowercase
    text = text.lower()

    text = re.sub(PUNCT_EXPR, repl, text)

    text = text.translate(TRANSLATE_TABLE)
    

    if stopwords is not None:
        text = ' '.join([w for w in tokenizer(text) if w not in stopwords])

    if lemmatize:
        text = ' '.join([NLTK_LEMMATIZER.lemmatize(w) for w in tokenizer(text)])

    text = re.sub(EXTRASPACE_EXPR, repl, text)

    return text

def load_tokens_from_text(txt_path: str, sep:str=',',strip_whitespace:bool=True):
    with open(txt_path, 'r') as f:
        text = f.read()
    text = text if not strip_whitespace else text.strip(' ')
    return list(set(text.split(sep)))
