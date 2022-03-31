import os
from numpy import float128, float32
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
from glove_utils import Glove
import torch.nn as nn
from utils import Similarity
import logging
from NTN_Model import LowRankNeuralTensorNetwork
from Eval_func import Hard_Similarity_eval, Hard_Similarity_Extention_eval, Transitive_eval
import argparse
import pickle 
import numpy as np
import torch.nn.functional as F
from HyperGraph import DocumentGraph
from Event_CL_Dataset import Event_CL_Dataset
import scipy.sparse as sp
from HyperGraph import target_event_Hypergraph_prepare
import sys
#此版本加入超图，Event_Glove_CL_1.py只有event和arg的emb 结合
def eval_model(ECL_model):
    ECL_model.eval()
    Hard_Similarity_eval(ECL_model)
    Hard_Similarity_Extention_eval(ECL_model)
    Transitive_eval(ECL_model)
    

class Parameter_Config:
    def __init__(self, args):
        self.epochs  = 10 
        self.learning_rate = args.lr
        self.batch_size = args.batch
        self.initial_accumulator_value = args.iav
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.do_train = True
        self.do_eval = True
        self.pool_type = "LowRankNeuralTensorNetwork"
 
                

           


class Event_CL(torch.nn.Module):
    def __init__(self, pc, pool_type = "LowRankNeuralTensorNetwork" ) -> None:
        super().__init__()
        self.Glove = Glove('./resource/glove.6B.100d.ext.txt')
        # self.embeddings = torch.nn.Embedding.from_pretrained(torch.tensor(self.Glove.embd))
        self.sim = Similarity()
        self.loss_fct = nn.CrossEntropyLoss()

        # self.if_train = pc.do_train
        # self.if_eval = pc.do_eval
        self.emb_dim = 100
        #加载composition模型
        if pool_type == "LowRankNeuralTensorNetwork":
            model_file = 'models/lowrank_r40_sigmoid_0.33_0.33_0.33_e9_hard_77.4.pt'
            embeddings = nn.Embedding(400000, self.emb_dim, padding_idx=1) # vocab_size,emb_dim
            self.composition_model = LowRankNeuralTensorNetwork(embeddings, k = 100, r = 40)# em_k, em_r
        if pool_type == "NTN":
            pass
        checkpoint = torch.load(model_file)
        if type(checkpoint) == dict:
            if 'event_model_state_dict' in checkpoint:
                state_dict = checkpoint['event_model_state_dict']
            else:
                state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        self.composition_model.load_state_dict(state_dict)
        self.compose_event_with_arg = nn.Linear(2 * self.emb_dim, self.emb_dim)
        #模型参数
        self.hard_negative_weight = 0
        self.device = pc.device
        
    def HyperGraph_init(self):
        #HyperGraph 

        self.HyperGraph_Model = DocumentGraph(self.train_vocab_dic) #len(self.vocab_dic)用来创建词典embeding，词都小写了
    def HyperGraphConstruction(self, batch_event):
        
        other_event_ids = []
        for event in batch_event:
            other_event_ids.append(self.dict_event_hyper[event][0])
        # reshape_other_event = []
        # for event in other_event:
        #     reshape_other_event.append(event.split(" "))
            
            
        inputs = other_event_ids
        
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
            num_edge = [i + len(self.dict_synset_to_event)  for i in num_edge]    
        
        max_n_edge = max(num_edge)

        max_se_len = max([len(i) for i in alias_inputs])#这一batch中所有doc，每个doc含有的最多的words数量（包含重复的）   《 ===》 和max_n_node区分

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
                    if u_input[s][i] == 0:#doc中的sent中的word
                        continue

                    rows.append(node_dic[idx][u_input[s][i]])
                    cols.append(s)
                    vals.append(1.0)
        
            if len(cols) == 0:
                s = 0
            else:
                s = max(cols) + 1
            if self.training:
                for i in node:
                    if i in self.arg_related_synset:#这个词id应该是在synset node包含的事件id里
                        temp = self.arg_related_synset[i]          
                        rows += [node_dic[idx][i]]*len(temp)#该doc中包含的keyword的id，len(temp)是该keyword与topic的关联个数
                        cols += [synset + s for synset in temp]#这把topic的id和该doc中词的id区分开了，但是不就会导致每个doc所对应的topic id不相同吗？
                        vals += [1.0]*len(temp)
            u_H = sp.coo_matrix((vals, (rows, cols)), shape=(max_n_node, max_n_edge))
            HT.append(np.asarray(u_H.T.todense()))
            
            alias_inputs[idx] = [j for j in range(max_n_node)]
            node_masks.append([1 for j in node] + (max_n_node - len(node)) * [0])
        
        return (alias_inputs, HT, items, node_masks)
    
    
    def HyperGraphRepresentation(self, HyerpGraph):
        alias_inputs, HT, items, node_masks = HyerpGraph[0], HyerpGraph[1], HyerpGraph[2], HyerpGraph[3]
        alias_inputs = torch.Tensor(alias_inputs).long().cuda()
        items = torch.Tensor(items).long().cuda()
        HT = torch.Tensor(HT).float().cuda()
        node_masks = torch.Tensor(node_masks).float().cuda()
        node = self.HyperGraph_Model(items, HT)
        get = lambda i: node[i][alias_inputs[i]]
        seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
        
        hidden = seq_hidden * node_masks.view(node_masks.shape[0], -1, 1).float()
        b = torch.sum(hidden * node_masks.view(node_masks.shape[0], -1, 1).float(),-2)/torch.sum(node_masks,-1).repeat(hidden.shape[2],1).transpose(0,1)          
        HGR = self.HyperGraph_Model.layer_normH(b)  
          
        return HGR
        
        
        
    def forward(self, z1, z2, z3):
        cos_sim = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))
        z1_z3_cos = self.sim(z1.unsqueeze(1), z3.unsqueeze(0))
        cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)
        labels = torch.arange(cos_sim.size(0)).long().to(self.device) #[batch]
        z3_weight = self.hard_negative_weight
        weights = torch.tensor(
            [[0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1)) + [0.0] * i + [z3_weight] + [0.0] * (z1_z3_cos.size(-1) - i - 1) for i in range(z1_z3_cos.size(-1))]
        ).to(self.device)
        cos_sim = cos_sim + weights
        loss = self.loss_fct(cos_sim, labels)
        
        return loss, cos_sim
        
    def pooler(self, event_emb, event_arg_embedding, pool_type = "LowRankNeuralTensorNetwork"):
        if pool_type == "add":
            return torch.sum(event_emb, 1)
        if pool_type == "LowRankNeuralTensorNetwork":
            if event_arg_embedding != None:
                subj_emb = event_emb.view(-1, event_emb.shape[2])[0:-1:3]
                verb_emb = event_emb.view(-1, event_emb.shape[2])[1:-1:3]
                obj_emb = event_emb.view(-1, event_emb.shape[2])[2::3] #如果是[2:-1:3] 取不到最后一个tensor
                event_emb = self.composition_model(subj_emb, verb_emb, obj_emb)
                event_emb = self.compose_event_with_arg(torch.cat((event_arg_embedding.squeeze(), event_emb),1))
            else:
                subj_emb = event_emb.view(-1, event_emb.shape[2])[0:-1:3]
                verb_emb = event_emb.view(-1, event_emb.shape[2])[1:-1:3]
                obj_emb = event_emb.view(-1, event_emb.shape[2])[2::3] #如果是[2:-1:3] 取不到最后一个tensor
                event_emb = self.composition_model(subj_emb, verb_emb, obj_emb)

            return event_emb
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='The parameter of The model.')
    #type是要传入的参数的数据类型  help是该参数的提示信息
    parser.add_argument('-lr', type=float128, default=5e-6, help='learning rate')
    parser.add_argument('-batch', type=int, default=64, help='batch size')
    parser.add_argument('-iav', type=float32, default=0.1, help='initial_accumulator_value')

    arg_str = ''
    args = parser.parse_args()
    for arg in vars(args):
        arg_str = arg_str + "_" + arg + "_" + str(getattr(args, arg))
    
    
    logging.basicConfig(filename='./log_Event_Glove_CL/Event_Glove_CL{}.log'.format(arg_str), format='%(asctime)s | %(levelname)s | %(message)s', level=logging.DEBUG, filemode='w') #有filename是文件日志输出,filemode是’w’的话，文件会被覆盖之前生成的文件会被覆盖

    seed = 42
    torch.manual_seed(seed)
    pc = Parameter_Config(args)
    

    ECL_model = Event_CL(pc, pool_type = pc.pool_type)
    ECL_model.to(pc.device)
    ECL_model.dict_event_hyper, ECL_model.arg_related_synset = target_event_Hypergraph_prepare(ECL_model)

    sys.exit()
    
    if pc.do_train:
        
        # Event_CL_dir = "../LDCCorpus/gigaword_eng_5/data/nyt_dataset_6" #对应Event_CL_nyt_dataset_6
        Event_CL_file = "./resource/triple_events/triple_events_0.txt"
        ECLD = Event_CL_Dataset(Event_CL_file)
        ECL_model.train_word_set = ECLD.word_set
        ECL_model.train_vocab_dic = ECLD.vocab_dic
        ECL_model.HyperGraph_init()
        
        
        
        train_dataloader = DataLoader(ECLD, batch_size = pc.batch_size, shuffle = True)
        optimizer = torch.optim.Adagrad(ECL_model.parameters(), lr = pc.learning_rate , initial_accumulator_value=pc.initial_accumulator_value)
        ECL_model.train()
        for epoch in range(pc.epochs):
            for batch, event_data in enumerate(train_dataloader):
                optimizer.zero_grad()#[x.grad for x in optimizer.param_groups[0]['params']] 没有梯度

                synset_node = event_data[0]

                #args embedding
                raw_event_arg = event_data[1][1]
                pos_event_arg = event_data[2][1]
                neg_event_arg = event_data[3][1]
                raw_event_arg_id = torch.tensor(list(map(ECL_model.Glove.transform, raw_event_arg))).to(pc.device) 
                pos_event_arg_id = torch.tensor(list(map(ECL_model.Glove.transform, pos_event_arg))).to(pc.device)
                neg_event_arg_id = torch.tensor(list(map(ECL_model.Glove.transform, neg_event_arg))).to(pc.device) 
                raw_event_arg_embeddings = ECL_model.composition_model.embeddings(raw_event_arg_id)#[batch, event_length, emb_size] [batch,1,emb_size]
                pos_event_arg_embeddings = ECL_model.composition_model.embeddings(pos_event_arg_id)
                neg_event_arg_embeddings = ECL_model.composition_model.embeddings(neg_event_arg_id)

                #event embeddings
                raw_event = event_data[1][0]
                pos_event = event_data[2][0]
                neg_event = event_data[3][0]
                
                raw_event_id = torch.tensor(list(map(ECL_model.Glove.transform, raw_event))).to(pc.device) #[batch, event_length] 一个event的三个词，event_length = 3
                pos_event_id = torch.tensor(list(map(ECL_model.Glove.transform, pos_event))).to(pc.device)
                neg_event_id = torch.tensor(list(map(ECL_model.Glove.transform, neg_event))).to(pc.device)
                raw_event_embeddings = ECL_model.composition_model.embeddings(raw_event_id)#[batch, event_length, emb_size] [batch,3,emb_size]
                pos_event_embeddings = ECL_model.composition_model.embeddings(pos_event_id)
                neg_event_embeddings = ECL_model.composition_model.embeddings(neg_event_id)
                
                #HyperGraph embeddings
                raw_event_hypergraph = ECL_model.HyperGraphConstruction(raw_event)
                pos_event_hypergraph = ECL_model.HyperGraphConstruction(pos_event)
                neg_event_hypergraph = ECL_model.HyperGraphConstruction(neg_event)
                
                
                #comprehensive embeddings
                pooler_raw_event_embeddings = ECL_model.pooler(raw_event_embeddings, raw_event_arg_embeddings, pool_type = pc.pool_type)
                pooler_pos_event_embeddings = ECL_model.pooler(pos_event_embeddings, pos_event_arg_embeddings, pool_type = pc.pool_type)
                pooler_neg_event_embeddings = ECL_model.pooler(neg_event_embeddings, neg_event_arg_embeddings, pool_type = pc.pool_type)
                
                
                loss, _ = ECL_model(pooler_raw_event_embeddings, pooler_pos_event_embeddings, pooler_neg_event_embeddings)
                loss.requires_grad_()
                loss.backward()
                optimizer.step()
                
                # for name, param in ECL_model.named_parameters():
                #     if param.requires_grad and name == "composition_model.embeddings.weight":
                #         print(param)
                if batch % 100 == 0:
                    loss, current = loss.item(), batch * len(event_data)
                    logging.info(f"loss: {loss:>7f}  [{current:>5d}/{pc.batch_size:>5d}]")
                    eval_model(ECL_model)
                    ECL_model.train()        
    if pc.do_eval:
        eval_model(ECL_model)  