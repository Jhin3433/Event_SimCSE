import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import os
import pickle
import random
import numpy as np
import scipy.sparse as sp
from glove_utils import Glove

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
            self.weight = Parameter(torch.Tensor(self.in_features, self.out_features))#[300,300]
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
        x_4att = x.matmul(self.weight2)#第一个HyperGraphAttentionLayerSparse x: [batch,max_node, initial_emb_size] self.weight2 [initial_emb_size, initial_emb_size]

        if self.transfer:
            x = x.matmul(self.weight)
            if self.bias is not None:
                x = x + self.bias        

        N1 = adj.shape[1] #number of edge
        N2 = adj.shape[2] #number of node

        pair = adj.nonzero().t()        

        get = lambda i: x_4att[i][adj[i].nonzero().t()[1]] #adj [batch, edge_num, max_node]
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
        x = self.gat1(x, H)#memory: 3G直接飙升到20G
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gat2(x, H)

        return x
    
class HyperGraph_Model(nn.Module):
    def __init__(self, Parameter_Config):
        super(HyperGraph_Model, self).__init__() #100
        # 参数配置
        self.hidden_size = Parameter_Config.hg_hidden_size
        self.if_Related_Hyper = Parameter_Config.if_Related_Hyper
        self.initial_feature = Parameter_Config.hg_initial_feature


        # 信息读取
        save_pickle = pickle.load(open("./resource/no_other_hyper/synset_event_graph_info.pickle","rb"))
        self.dic_event_to_synset = save_pickle[0]
        self.dict_synset_to_event = save_pickle[1]
        self.vocab_dic = save_pickle[2]
        self.dict_synset_index = save_pickle[3]#raw_event的所有synset，以及其doc涉及的verb synset
        del save_pickle


        
        #Glove词典和 训练集词典对齐, https://blog.csdn.net/weixin_37763484/article/details/114274851
        self.n_node = len(self.vocab_dic)
        self.Glove = Glove('./resource/glove.6B.100d.ext.txt')

        self.dropout = Parameter_Config.hg_dropout
        self.initial_feature = Parameter_Config.hg_initial_feature
        # self.normalization = False
        self.layer_normH = nn.LayerNorm(self.hidden_size, eps=1e-6)
        # if self.normalization:
        #     self.layer_normC = nn.LayerNorm(self.n_categories, eps=1e-6)
        
        
        self.reset_parameters()
        self.hgnn = HGNN_ATT(self.initial_feature, self.initial_feature, self.hidden_size, dropout = self.dropout)

        num = 0
        emb_randn = torch.randn(self.n_node+1, self.initial_feature)
        for word in self.Glove.vocab_id:
                if word in self.vocab_dic:
                    emb_randn[self.vocab_dic[word]] = torch.FloatTensor(self.Glove.embd[self.Glove.id(word), :])
                    num += 1

        self.Hypergraph_embedding = nn.Embedding.from_pretrained(emb_randn)
        # self.Hypergraph_embedding = nn.Embedding(self.n_node+1, self.initial_feature, padding_idx=0)#nn.init.xavier_normal_

        del self.Glove


        

        




    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
            
    def forward(self, HyerpGraph):

        alias_inputs, HT, items, node_masks = HyerpGraph[0], HyerpGraph[1], HyerpGraph[2], HyerpGraph[3]
        alias_inputs = torch.Tensor(alias_inputs).long().cuda()
        items = torch.Tensor(items).long().cuda()
        HT = torch.Tensor(HT).float().cuda()
        node_masks = torch.Tensor(node_masks).float().cuda()#3.1G
    
        hidden = self.Hypergraph_embedding(items)#  model.py/node = model(items, HT)#[8,max_len,100]
        nodes = self.hgnn(hidden, HT)

        get = lambda i: nodes[i][alias_inputs[i]]
        seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
        
        hidden = seq_hidden * node_masks.view(node_masks.shape[0], -1, 1).float()#model.py/compute_scores
        b = torch.sum(hidden * node_masks.view(node_masks.shape[0], -1, 1).float(),-2)/torch.sum(node_masks,-1).repeat(hidden.shape[2],1).transpose(0,1)          
        HGR = self.layer_normH(b)  
          
        return HGR

    def Hyper_transform_to_id(self, event):
        event_id = []
        for arg in event.split(" "):
                # assert arg in self.vocab_dic
            if arg in self.vocab_dic:
                event_id.append(self.vocab_dic[arg])
            else:
                event_id.append(0)
            # if arg in self.vocab_dic:#因为超图里的event是从全图取的，所以有一部分节点没在训练集
            #     event_id.append(self.vocab_dic[arg])
            # else:
            #     event_id.append(0)
        return event_id    


    def HyperGraphConstruction(self, event_hypers):
        # other_event_ids = []
        # for event in batch_event:
        #     other_event = []
        #     other_event.append(self.Hyper_transform_to_id(event)) 


        #     # other_event_synset = []
        #     # if event in ECL_model.dic_event_to_synset:
        #     synsets_list = self.dic_event_to_synset[event]
        #     for arg_synset in synsets_list:
        #         synset = arg_synset[1]
        #         if synset.find(".v.") == -1:
        #             continue
        #         for arg_event in self.dict_synset_to_event[synset]:
        #             # other_event.append(list(ECL_model.Glove.transform(arg_event[1])))
        #             other_event.append( self.Hyper_transform_to_id(arg_event[1])) 
        #             arg_id = self.Hyper_transform_to_id(arg_event[0])[0]
        #             if arg_id not in self.arg_related_synset: #and arg_id != 1:
        #                 self.arg_related_synset[arg_id] = self.dict_synset_index[synset]
        #     random.shuffle(other_event)#不用返回值，直接对参数变量进行修改
        #     other_event_ids.append(other_event[0:10] if len(other_event) > 10 else other_event)
        if self.if_Related_Hyper:
            arg_related_synset = {}
            other_event_ids = []
            for event_hyper in event_hypers:
                other_event = []
                for event in event_hyper.split(","):
                    other_event.append(self.Hyper_transform_to_id(event))
            other_event_ids.append(other_event) 

            #这儿还没创建hyper，需要过滤other event verb related synset



        else:
            arg_related_synset = {}
            events = event_hypers
            other_event_ids = []
            for event in events:
                other_event = []
                other_event.append(self.Hyper_transform_to_id(event))

                if self.training:
                    synset_list = self.dic_event_to_synset[event]
                    for arg_synset in synset_list:
                        assert arg_synset[1] in self.dict_synset_index
                        arg_id = self.Hyper_transform_to_id(arg_synset[0])[0]
                        if arg_synset[0] not in arg_related_synset:
                            arg_related_synset[arg_id] = set()
                            arg_related_synset[arg_id].add(self.dict_synset_index[arg_synset[1]])
                        else:
                            arg_related_synset[arg_id].add(self.dict_synset_index[arg_synset[1]])

                other_event_ids.append(other_event)

        inputs = other_event_ids
        #------------------------------------------------对这一batch的event进行统计------------------------------------------------
        #构造超图
        items, n_node, HT, alias_inputs, node_masks, node_dic = [], [], [], [], [], []
        for u_input in inputs:#input为一个doc
            temp_s = []#doc包含的所有词
            for s in u_input:#s为一个sent
                temp_s += s
            
            temp_l = list(set(temp_s))    #该doc中无重复的所有词
            temp_dic = {temp_l[i]: i for i in range(len(temp_l))}    #templ中的词id按顺序index    
            n_node.append(temp_l)
            alias_inputs.append([temp_dic[i] for i in temp_s])#[8,] 把temp_s中词id 转换为temp_dic中的id，把doc中sent的词堆到一个list上 
            node_dic.append(temp_dic)


        max_n_node = np.max([len(i) for i in n_node])#这一batch中所有doc，含有最多的无重复词的数量
        num_edge = [len(i) for i in inputs]#这一batch中所有doc，含有句子数
        if self.training:
            num_edge = [i + len(self.dict_synset_index)  for i in num_edge]    
        
        max_n_edge = max(num_edge)

        max_se_len = max([len(i) for i in alias_inputs])#这一batch中所有doc，每个doc含有的最多的words数量（包含重复的）   《 ===》 和max_n_node区分

        #------------------------------------------------对这一batch的event进行处理------------------------------------------------
        for idx in range(len(inputs)):
            #以下为对于一篇doc的处理
            u_input = inputs[idx]#该doc包含的sent，每个sent含有词id
            node = n_node[idx]#该doc中无重复的所有词id
            items.append(node + (max_n_node - len(node)) * [0])#将node补0，使得node长度变为max_n_node


            rows = []#该doc中所有词id（node_dic中的）
            cols = []#和和句子有关联
            vals = []
            
            
            for s in range(len(u_input)):
                for i in np.arange(len(u_input[s])):
                    # if u_input[s][i] == 0:#doc中的sent中的word
                    #     continue

                    rows.append(node_dic[idx][u_input[s][i]])
                    cols.append(s)
                    vals.append(1.0)
        
            if len(cols) == 0:
                s = 0
            else:
                s = max(cols) + 1
            if self.training:
                for i in node:
                    if i in arg_related_synset:#这个词id应该是在synset node包含的事件id里
                        temp = list(arg_related_synset[i])#因为一个词只与一个synset有关系，所以变成了列表，          
                        rows += [node_dic[idx][i]]*len(temp)#该doc中包含的keyword的id，len(temp)是该keyword与topic的关联个数
                        cols += [synset + s for synset in temp]#这把topic的id和该doc中词的id区分开了，但是不就会导致每个doc所对应的topic id不相同吗？
                        vals += [1.0]*len(temp)
            u_H = sp.coo_matrix((vals, (rows, cols)), shape=(max_n_node, max_n_edge))
            HT.append(np.asarray(u_H.T.todense()))
            
            alias_inputs[idx] = [j for j in range(max_n_node)]
            node_masks.append([1 for j in node] + (max_n_node - len(node)) * [0])
        
        return (alias_inputs, HT, items, node_masks)
    
        

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
                    if arg_id not in arg_related_synset and arg_id != 1:
                        arg_related_synset[arg_id] = dict_synset_index[synset]
                    
                    # other_event_synset += [synset] * len(other_event)
                    
            random.shuffle(other_event)#不用返回值，直接对参数变量进行修改
            dict_event_hyper[event] = other_event[0:100] if len(other_event) > 100 else other_event
        
        pickle.dump(dict_event_hyper, open('resource/dict_event_hyper.pickle', 'wb'))
        pickle.dump(arg_related_synset, open('resource/arg_related_synset.pickle', 'wb'))
        print("dict_event_hyper.pickle is saved well.")
        print("arg_related_synset.pickle is saved well.")

    del dic_event_to_synset
    del dict_synset_to_event
    return dict_event_hyper, arg_related_synset
    
    