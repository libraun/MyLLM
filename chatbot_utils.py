import re

# Match anything after a "Reference" (or "External Links") header
ENDTEXT_EXPR = re.compile("==+ +?[References|External Links][\\S\\s]*")

# Match common JavaScript and header plaintext tags
WIKITAG_EXPR = re.compile("==+[^=]*?==+|\\n|\\r|{[^}]*?}")

# Match more than one space
EXTRASPACE_EXPR = re.compile("  +")

def preprocess(text: str, repl: str=" ") -> str:

    # Convert text to lowercase
    text = text.lower()

    # Remove wikipedia-specific tags/unnecessary text
    text = re.sub(ENDTEXT_EXPR, repl, text)
    text = re.sub(WIKITAG_EXPR, repl, text)

    # Remove all extra spaces and non-unicode characters
    text = re.sub(EXTRASPACE_EXPR, repl, text)
    #text = re.sub(BADCHAR_EXPR, repl, text)

    return text



import json
from typing import List
from collections import Counter, OrderedDict
from torchtext.vocab import vocab
from torchtext.data.utils import get_tokenizer

class EmbeddingRetriever:
    
    def __init__(self, 
                 init_corpus: List[str], 
                 tokenizer_name: str="spacy",
                 tokenizer_language: str="en_core_web_sm",
                 specials: List[str] = ["<unk>","<pad>","<bos>","<eos>"]):

        self.tokenizer = get_tokenizer(tokenizer_name, 
                                       language=tokenizer_language)

        self.__init_vocab__(init_corpus, specials=specials)



    def __init_vocab__(self, corpus, specials):
        
        counter = Counter()
        for text in corpus:
            tokens = self.tokenizer(text)
            counter.update(tokens)

        sorted_by_freq_tuples = sorted(counter.items(), 
                                       key=lambda x: x[1], 
                                       reverse=True)
        
        ordered_dict = OrderedDict(sorted_by_freq_tuples)    
        self.vocab = vocab(ordered_dict, 
                           specials=specials,
                           specials_first=True)
        
    def save_vocab(self, path: str) -> int:

        obj = json.dumps(self.vocab)
        
        with open(path, "w+") as outfile:
            bytes_written = outfile.write(obj)
        
        return bytes_written
        


        


    
    







        
     
      
      


