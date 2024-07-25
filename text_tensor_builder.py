import pickle

import torch
import torchtext

torchtext.disable_torchtext_deprecation_warning()

from typing import List
from collections import Counter, OrderedDict
from torchtext.vocab import vocab
from torchtext.data.utils import get_tokenizer

class TextTensorBuilder:

    tokenizer = get_tokenizer("spacy","en_core_web_sm")
    def __init__(self, corpus,
                 specials: List[str],
                 default_index: int = 0,
                 save_filepath: str=None,
                 min_freq: int=5,
                 tokenizer_name: str="spacy", 
                 tokenizer_version: str="en_core_web_sm",):
        
        self.tokenizer = get_tokenizer(tokenizer_name,tokenizer_version)
        self.__build_vocab__(corpus, specials, default_index, save_filepath, min_freq=min_freq)

    def text_to_tensor(self, doc: str | List[str], 
                       tokenize: bool=True,
                       reverse_tokens: bool=False,
                       remove_idx: int = 0 ) -> torch.Tensor: 
        
        tokens = doc if not tokenize else self.tokenizer(doc)

        # Optionally reverse input sequence (trusting the paper)
        if reverse_tokens:
            tokens.reverse()
        
        text_tensor = [self.lang_vocab[token] for token in tokens if self.lang_vocab[token] != remove_idx]
        text_tensor = torch.tensor(text_tensor, dtype=torch.long)

        return text_tensor
    
    def __build_vocab__(self, corpus: List[str], 
                        specials: List[str],
                        default_index: int = 0, 
                        save_filepath: str=None, 
                        min_freq: int=5):
        counter = Counter()
        for text in corpus:
            tokens = self.tokenizer(text)
            counter.update(tokens)

        sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        
        ordered_dict = OrderedDict(sorted_by_freq_tuples)    
        result = vocab(ordered_dict, min_freq=min_freq, specials=specials)

        result.set_default_index(default_index)

        self.lang_vocab = result

        if save_filepath is not None:
            self.__save_vocab__(save_filepath)
    

    def __save_vocab__(self, filename: str):

        with open(filename, "wb+") as f:
            pickle.dump(self.lang_vocab, f)