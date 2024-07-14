import torch
import torchtext

torchtext.disable_torchtext_deprecation_warning()

import pickle

from typing import List
from collections import Counter, OrderedDict
from torchtext.vocab import vocab

from torchtext.data.utils import get_tokenizer

class TextTensorBuilder:

    tokenizer = get_tokenizer("spacy","en_core_web_sm")
    
    @classmethod
    def convert_text_to_tensor(cls, vocab_dict,
                               doc: str | List[str], 
                               tokenize: bool=True ) -> torch.Tensor: 
        if tokenize:
            tokens = cls.tokenizer(doc)
        else:
            tokens = doc
        text_tensor = [vocab_dict[token] for token in tokens]
        text_tensor = torch.tensor(text_tensor, dtype=torch.long)

        return text_tensor
    
    @classmethod
    def build_vocab(cls,corpus: List[str],
                    specials: List[str]=["<UNK_IDX>","<PAD_IDX>","<BOS_IDX>","<EOS_IDX>"],
                    default_token:str = "<UNK_IDX>"):
        counter = Counter()
        for text in corpus:
            tokens = cls.tokenizer(text)
            counter.update(tokens)

        sorted_by_freq_tuples = sorted(counter.items(), 
                                       key=lambda x: x[1], 
                                       reverse=True)
        
        ordered_dict = OrderedDict(sorted_by_freq_tuples)    
        result = vocab(ordered_dict, specials=specials)

        result.set_default_index(result[default_token])

        return result
    
    @classmethod
    def tokenize(cls, text):
        return cls.tokenizer(text)
    
    @staticmethod
    def save_vocab(vocab, path) -> None:
        with open(path,"wb+") as f:
            pickle.dump(vocab, f)

