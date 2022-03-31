import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import os
import pickle


class HyperGraphAttentionLayerSparse(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha, transfer, concat=True, bias=False):
        super(HyperGraphAttentionLayerSparse, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.transfer = transfer

        if self.transfer:
            self.weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        else:
            self.register_parameter('weight', None)

        self.weight2 = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.weight3 = Parameter(torch.Tensor(self.out_features, self.out_features))

        if bias:
            self.bias = Parameter(torch.Tensor(self.out_features))
        else:
            self.register_parameter('bias', None)

        self.word_context = nn.Embedding(1, self.out_features)
      
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))   
        self.a2 = nn.Parameter(torch.zeros(size=(2*out_features, 1)))        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)
        self.weight2.data.uniform_(-stdv, stdv)
        self.weight3.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        
        nn.init.uniform_(self.a.data, -stdv, stdv)
        nn.init.uniform_(self.a2.data, -stdv, stdv)
        nn.init.uniform_(self.word_context.weight.data, -stdv, stdv)


    def forward(self, x, adj):
        x_4att = x.matmul(self.weight2)

        if self.transfer:
            x = x.matmul(self.weight)
            if self.bias is not None:
                x = x + self.bias        

        N1 = adj.shape[1] #number of edge
        N2 = adj.shape[2] #number of node

        pair = adj.nonzero().t()        

        get = lambda i: x_4att[i][adj[i].nonzero().t()[1]]
        x1 = torch.cat([get(i) for i in torch.arange(x.shape[0]).long()])


        q1 = self.word_context.weight[0:].view(1, -1).repeat(x1.shape[0],1).view(x1.shape[0], self.out_features)
        
        pair_h = torch.cat((q1, x1), dim=-1)
        pair_e = self.leakyrelu(torch.matmul(pair_h, self.a).squeeze()).t()
        assert not torch.isnan(pair_e).any()
        pair_e = F.dropout(pair_e, self.dropout, training=self.training)

        e = torch.sparse_coo_tensor(pair, pair_e, torch.Size([x.shape[0], N1, N2])).to_dense()

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)


        attention_edge = F.softmax(attention, dim=2)

        edge = torch.matmul(attention_edge, x)
        
        edge = F.dropout(edge, self.dropout, training=self.training)

        edge_4att = edge.matmul(self.weight3)

        get = lambda i: edge_4att[i][adj[i].nonzero().t()[0]]
        y1 = torch.cat([get(i) for i in torch.arange(x.shape[0]).long()])

        get = lambda i: x_4att[i][adj[i].nonzero().t()[1]]
        q1 = torch.cat([get(i) for i in torch.arange(x.shape[0]).long()])

        pair_h = torch.cat((q1, y1), dim=-1)
        pair_e = self.leakyrelu(torch.matmul(pair_h, self.a2).squeeze()).t()
        assert not torch.isnan(pair_e).any()
        pair_e = F.dropout(pair_e, self.dropout, training=self.training)

        e = torch.sparse_coo_tensor(pair, pair_e, torch.Size([x.shape[0], N1, N2])).to_dense()

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)

        attention_node = F.softmax(attention.transpose(1,2), dim=2)

        node = torch.matmul(attention_node, edge)


        if self.concat:
            node = F.elu(node)

        return node

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
    
    
class HGNN_ATT(nn.Module):
    def __init__(self, input_size, n_hid, output_size, dropout=0.3):#input_size 300, n_hid 300, output_size 100
        super(HGNN_ATT, self).__init__()
        self.dropout = dropout#0.3
        self.gat1 = HyperGraphAttentionLayerSparse(input_size, n_hid, dropout=self.dropout, alpha=0.2, transfer = False, concat=True)
        self.gat2 = HyperGraphAttentionLayerSparse(n_hid, output_size, dropout=self.dropout, alpha=0.2, transfer = True, concat=False)
        
    def forward(self, x, H):   
        x = self.gat1(x, H)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gat2(x, H)

        return x
    
class DocumentGraph(nn.Module):
    def __init__(self, vocab_dic):
        super(DocumentGraph, self).__init__() #100
        self.hidden_size = 100
        self.n_node = len(vocab_dic)
        # self.n_categories = len(dict_synset_to_event)
        self.dropout = 0.3
        self.initial_feature = 300
        # self.normalization = False
        self.Hypergraph_embedding = nn.Embedding(self.n_node+1, self.initial_feature, padding_idx=0)#(9335,300)?
        self.layer_normH = nn.LayerNorm(self.hidden_size, eps=1e-6)
        # if self.normalization:
        #     self.layer_normC = nn.LayerNorm(self.n_categories, eps=1e-6)
        self.reset_parameters()
        
        
        self.hgnn = HGNN_ATT(self.initial_feature, self.initial_feature, self.hidden_size, dropout = self.dropout)
        
        
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
            
    def forward(self, inputs, HT):
        
        hidden = self.Hypergraph_embedding(inputs)
        nodes = self.hgnn(hidden, HT)
        return nodes




def target_event_Hypergraph_prepare(ECL_model):
    
    if os.path.exists("./resource/dict_event_hyper.pickle") and os.path.exists("./resource/arg_related_synset.pickle"):
        dict_event_hyper = pickle.load(open("./resource/dict_event_hyper.pickle","rb"))
        arg_related_synset = pickle.load(open("./resource/arg_related_synset.pickle","rb"))
    else:
        dic_event_to_synset = pickle.load(open("./resource/dict_event_to_synset.pickle","rb"))
        dict_synset_to_event = pickle.load(open("./resource/dict_synset_to_event.pickle","rb"))

        all_synset = set()
        for synset in dict_synset_to_event:
            all_synset.add(synset)
            
        dict_synset_index = {}
        for i in all_synset:
            dict_synset_index[i] = len(dict_synset_index) + 1
            

        dict_event_hyper = {}
        arg_related_synset = {}#类比分类中的self.keywords

        for event in dic_event_to_synset:
            other_event = []
            # other_event_synset = []
            # if event in ECL_model.dic_event_to_synset:
            synsets_list = dic_event_to_synset[event]
            for arg_synset in synsets_list:
                synset = arg_synset[1]
                for arg_event in dict_synset_to_event[synset]:
                    other_event.append(list(ECL_model.Glove.transform(arg_event[1])))
                    
                    arg_id = ECL_model.Glove.transform(arg_event[0])[0]
                    if arg_id not in arg_related_synset:
                        arg_related_synset[arg_id] = dict_synset_index[synset]
                    
                    # other_event_synset += [synset] * len(other_event)
            dict_event_hyper[event] = other_event
        
        pickle.dump(dict_event_hyper, open('resource/dict_event_hyper.pickle', 'wb'))
        pickle.dump(arg_related_synset, open('resource/arg_related_synset.pickle', 'wb'))
        print("dict_event_hyper.pickle is saved well.")
        print("arg_related_synset.pickle is saved well.")

    del dic_event_to_synset
    del dict_synset_to_event
    return dict_event_hyper, arg_related_synset
    
    