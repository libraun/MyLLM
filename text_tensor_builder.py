import pickle

import torch
import torchtext.vocab

torchtext.disable_torchtext_deprecation_warning()

from typing import List
from collections import Counter, OrderedDict
from torchtext.vocab import vocab
from torchtext.data.utils import get_tokenizer

# TextTensorProvider provides utilities for building tensors from text,
# and provides an easy way to create torchtext vocab objects.
class TextTensorBuilder:

    tokenizer = get_tokenizer("spacy", "en_core_web_sm")

    # Accepts a torchtext vocabulary object for token lookups, and a string 
    # to be parsed to tensor (using vocab)
    @classmethod
    def text_to_tensor(cls, lang_vocab,
                       doc: str | List[str], 
                       max_tokens: int = None,
                       reverse_tokens: bool=False,
                       tokenize: bool=True) -> torch.Tensor: 
        
        tokens = doc if not tokenize else cls.tokenizer(doc)
        
        if max_tokens is not None:
            tokens = tokens[ : max_tokens]

        # Optionally reverse input sequence (trusting the paper)
        if reverse_tokens:
            tokens.reverse()
        
        text_tensor = [lang_vocab[token] for token in tokens]
        text_tensor = torch.tensor(text_tensor, dtype=torch.long)

        return text_tensor
    
    @classmethod
    def build_vocab(cls, corpus: List[str], 
                    specials: List[str],
                    default_index_token: str ="<UNK>", 
                    save_filepath: str=None, 
                    min_freq: int=5):
        
        # Apply tokenizer to each entry in corpus
        tokenized_entries = iter(map(cls.tokenizer, corpus))

        counter = Counter(tokenized_entries)
        sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        
        ordered_dict = OrderedDict(sorted_by_freq_tuples)    
        vocab_object = vocab(ordered_dict, min_freq=min_freq, specials=specials)

        default_idx = vocab_object[default_index_token]

        # Sets the default index for lookups
        vocab_object.set_default_index(default_idx)

        if save_filepath is not None:
            cls.save_vocab(vocab_object, save_filepath)

        return vocab_object
    
    @classmethod
    def save_vocab(cls, lang_vocab, filename: str):

        with open(filename, "wb+") as f:
            pickle.dump(lang_vocab, f)