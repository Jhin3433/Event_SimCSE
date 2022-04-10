from glove_utils import Glove

import torch.nn as nn
import torch

Glove = Glove('./resource/glove.6B.100d.ext.txt')
embeddings = torch.nn.Embedding.from_pretrained(torch.tensor(self.Glove.embd)) 


Hypergraph_embedding = nn.Embedding(self.n_node+1, self.initial_feature, padding_idx=0)#(9335,300)?