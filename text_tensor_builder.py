import torch
import torch.nn as nn
import torchtext

torchtext.disable_torchtext_deprecation_warning()

from typing import List
from collections import Counter, OrderedDict
from torchtext.vocab import vocab
from torchtext.data.utils import get_tokenizer

class TextTensorBuilder:

    def __init__(self, 
                 embedding_dim: int, 
                 data: List[str],
                 tokenizer: str="spacy",
                 tokenizer_lang: str="en_core_web_sm",
                 specials: List[str]=["<unk>", "<pad>", "<bos>", "<eos>", "<BEGINDOC>","<ENDDOC>"],
                 default_token: str="<unk>"):
        
        if default_token not in specials:
            specials.insert(0, default_token)

        self.tokenizer = get_tokenizer(tokenizer,tokenizer_lang)
        self.en_vocab = self.__build_vocab__(data, specials)

        self.en_vocab.set_default_index(self.en_vocab[default_token])

        self.embedding = nn.Embedding(len(self.en_vocab), embedding_dim,
                                      padding_idx=self.en_vocab["<pad>"])
    
    def convert_text_to_tensor(self, 
                               doc: str | List[str], 
                               tokenize: bool=True ) -> torch.Tensor: 
        if tokenize:
            tokens = self.tokenizer(doc)
        else:
            tokens = doc
        text_tensor = [self.en_vocab[token] for token in tokens]
        text_tensor = torch.tensor(text_tensor, dtype=torch.long)

        return text_tensor
    
    def get_vocab(self):
        return self.en_vocab

    def __build_vocab__(self,
                        corpus: List[str],
                        specials: List[str]):
        counter = Counter()
        for text in corpus:
            tokens = self.tokenizer(text)
            counter.update(tokens)

        sorted_by_freq_tuples = sorted(counter.items(), 
                                       key=lambda x: x[1], 
                                       reverse=True)
        
        ordered_dict = OrderedDict(sorted_by_freq_tuples)    
        result = vocab(ordered_dict, specials=specials)

        return result