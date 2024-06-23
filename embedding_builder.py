import torch
import torch.nn as nn
import torchtext

torchtext.disable_torchtext_deprecation_warning()

from typing import List
from collections import Counter, OrderedDict
from torchtext.vocab import vocab
from torchtext.data.utils import get_tokenizer

class EmbeddingBuilder:

    def __init__(self, 
                 embedding_dim: int, 
                 data: List[str],
                 tokenizer: str="spacy",
                 tokenizer_lang: str="en_core_web_sm",
                 specials: List[str]=["<unk>", "<pad>", "<bos>", "<eos>"]):

        self.tokenizer = get_tokenizer(tokenizer,tokenizer_lang)
        self.en_vocab = self.__build_vocab__(data, specials)

        self.embedding = nn.Embedding(len(self.en_vocab), 
                                      embedding_dim,
                                      padding_idx=self.en_vocab["<pad>"])
    
    def get_embeddings(self, document: str) -> None:
    
        tokens = self.tokenizer(document)

        result = [vocab[word] for word in tokens]
        result = torch.tensor(result, dtype=torch.long)

        result = self.embedding(result)
        return result

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