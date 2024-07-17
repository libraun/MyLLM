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
    
    @classmethod
    def text_to_tensor(self, lang_vocab,
                       doc: str | List[str], 
                       tokenize: bool=True ) -> torch.Tensor: 
        
        tokens = doc if not tokenize else self.tokenizer(doc)
        
        text_tensor = [lang_vocab[token] for token in tokens]
        text_tensor = torch.tensor(text_tensor, dtype=torch.long)

        return text_tensor

    @classmethod
    def build_vocab(cls,corpus: List[str], specials: List[str]):
        counter = Counter()
        for text in corpus:
            tokens = cls.tokenizer(text)
            counter.update(tokens)

        sorted_by_freq_tuples = sorted(counter.items(), 
                                       key=lambda x: x[1], 
                                       reverse=True)
        
        ordered_dict = OrderedDict(sorted_by_freq_tuples)    
        result = vocab(ordered_dict, specials=specials)

        return result
    
    @staticmethod
    def save_vocab(lang_vocab, filename: str):

        with open(filename, "wb+") as f:
            pickle.dump(lang_vocab)