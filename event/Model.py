
import torch
from NTN_Model import LowRankNeuralTensorNetwork
from utils import Similarity
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from HyperGraph import HyperGraph_Model

class Event_CL(torch.nn.Module):
    def __init__(self, pc, pool_type = "LowRankNeuralTensorNetwork" ) -> None:
        super().__init__()
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
        self.compose_event_with_hyper = nn.Linear(pc.hg_hidden_size + self.emb_dim, self.emb_dim)
        #模型参数
        self.hard_negative_weight = 0
        self.device = pc.device

    def HyperGraph_init(self, Parameter_Config):
        #HyperGraph 
        self.HyperGraph_Model = HyperGraph_Model(Parameter_Config).to(self.device) #len(self.vocab_dic)用来创建词典embeding，词都小写了。  这儿可以调整参数


        
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
        
    def pooler(self, event_emb, event_arg_embedding = None, event_hypergraph = None, pool_type = "LowRankNeuralTensorNetwork"):
        if pool_type == "add":
            return torch.sum(event_emb, 1)
        if pool_type == "LowRankNeuralTensorNetwork":
            if event_arg_embedding != None and event_hypergraph != None:
                subj_emb = event_emb.view(-1, event_emb.shape[2])[0:-1:3]
                verb_emb = event_emb.view(-1, event_emb.shape[2])[1:-1:3]
                obj_emb = event_emb.view(-1, event_emb.shape[2])[2::3] #如果是[2:-1:3] 取不到最后一个tensor
                event_emb = self.composition_model(subj_emb, verb_emb, obj_emb)
                event_emb = self.compose_event_with_arg(torch.cat((event_arg_embedding.squeeze(), event_emb),1))
                event_emb = self.compose_event_with_hyper(torch.cat((event_hypergraph.squeeze(), event_emb),1))


            else:
                subj_emb = event_emb.view(-1, event_emb.shape[2])[0:-1:3]
                verb_emb = event_emb.view(-1, event_emb.shape[2])[1:-1:3]
                obj_emb = event_emb.view(-1, event_emb.shape[2])[2::3] #如果是[2:-1:3] 取不到最后一个tensor
                event_emb = self.composition_model(subj_emb, verb_emb, obj_emb)

            return event_emb