import os
from numpy import float128, float32
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import logging
from Eval_func import Hard_Similarity_only_hyper_eval
import argparse

from Event_CL_Dataset import Event_CL_Dataset
from NTN_Model import LowRankNeuralTensorNetwork
from glove_utils import Glove
from utils import Similarity
import torch.nn as nn
from HyperGraph import HyperGraph_Model

#此版本只有超图
def eval_model(EH_model):
    EH_model.eval()
    Hard_Similarity_only_hyper_eval(EH_model)
    # Hard_Similarity_Extention_eval(EH_model)
    # Transitive_eval(EH_model)


class Parameter_Config:
    def __init__(self, args):
        self.epochs  = 10 
        self.learning_rate = args.lr
        self.batch_size = args.batch
        self.initial_accumulator_value = args.iav
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.do_train = args.do_train
        self.do_eval = args.do_eval

 
        self.if_Related_Hyper = args.if_doc_hyper
        # HyperGraph_Model Parameters
        self.hg_hidden_size = 100
        self.hg_dropout = 0.1
        self.hg_initial_feature = 100 #词的embd size



class Event_Hyper(torch.nn.Module):
    def __init__(self, pc) -> None:
        super().__init__()
        self.sim = Similarity()
        self.loss_fct = nn.CrossEntropyLoss()

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
        
 


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='The parameter of The model.')
    #type是要传入的参数的数据类型  help是该参数的提示信息
    parser.add_argument('-lr', type=float128, default=5e-6, help='learning rate')
    parser.add_argument('-batch', type=int, default=64, help='batch size')
    parser.add_argument('-iav', type=float32, default=0.1, help='initial_accumulator_value')
    parser.add_argument('-if_doc_hyper', action='store_true', help='if use events doc to construct hyper') # type = bool ❌
    parser.add_argument('-do_train', action='store_true', help='if training') 
    parser.add_argument('-do_eval', action='store_true', help='if eval') 

    parser.add_argument('-hg_hidden_size', type=int, default=100, help='hyper model hidden size.') 
    parser.add_argument('-hg_dropout', type=float32, default=0.3, help='hyper model dropout.') 
    parser.add_argument('-hg_initial_feature', type=int, default=100, help='hyper model initial_feature.') 

    arg_str = ''
    args = parser.parse_args()
    for arg in vars(args):
        arg_str = arg_str + "_" + arg + "_" + str(getattr(args, arg))
    
    
    logging.basicConfig(filename='./log_Event_Glove_CL/Only_Hyper/Event_Glove_Only_Hyper{}.log'.format(arg_str), format='%(asctime)s | %(levelname)s | %(message)s', level=logging.DEBUG, filemode='w') #有filename是文件日志输出,filemode是’w’的话，文件会被覆盖之前生成的文件会被覆盖



    seed = 42
    torch.manual_seed(seed)
    pc = Parameter_Config(args)
    

    EH_model = Event_Hyper(pc)#memory:1.4G
    EH_model.to(pc.device)
    # ECL_model.dict_event_hyper, ECL_model.arg_related_synset = target_event_Hypergraph_prepare(ECL_model)


    if pc.do_train:
        
        Event_CL_file = "./resource/no_other_hyper/hard_hyper_verb_centric_triple_events.txt"
        if pc.if_Related_Hyper:
            Hyper_file = "./resource/hard_triple_events/hyper_events_0.txt"
            ECLD = Event_CL_Dataset(Event_CL_file, Hyper_file = Hyper_file)   
        else:  
            ECLD = Event_CL_Dataset(Event_CL_file) 



        EH_model.HyperGraph_init(pc)#memory:1.6G
        
        
        train_dataloader = DataLoader(ECLD, batch_size = pc.batch_size, shuffle = True)
        optimizer = torch.optim.Adagrad(EH_model.parameters(), lr = pc.learning_rate , initial_accumulator_value=pc.initial_accumulator_value)
        EH_model.train()
        for epoch in range(pc.epochs):
            for batch, event_data in enumerate(train_dataloader):
                optimizer.zero_grad()#[x.grad for x in optimizer.param_groups[0]['params']] 没有梯度

                synset_node = event_data[0]


                #HyperGraph embeddings

                if pc.if_Related_Hyper:
                    raw_event_hyper = event_data[4]
                    pos_event_hyper = event_data[5]
                    neg_event_hyper = event_data[6]

                else:
                    raw_event_hyper = event_data[1][0]
                    pos_event_hyper = event_data[2][0]
                    neg_event_hyper = event_data[3][0]


                raw_event_hypergraph = EH_model.HyperGraph_Model(EH_model.HyperGraph_Model.HyperGraphConstruction(raw_event_hyper))
                pos_event_hypergraph = EH_model.HyperGraph_Model(EH_model.HyperGraph_Model.HyperGraphConstruction(pos_event_hyper))
                neg_event_hypergraph = EH_model.HyperGraph_Model(EH_model.HyperGraph_Model.HyperGraphConstruction(neg_event_hyper))
                loss, _ = EH_model(raw_event_hypergraph, pos_event_hypergraph, neg_event_hypergraph)
                # loss.requires_grad_()
                loss.backward()
                optimizer.step()

                if batch % 10 == 0:
                    loss, current = loss.item(), batch * len(event_data)
                    logging.info(f"loss: {loss:>7f}  [{current:>5d}/{pc.batch_size:>5d}]")
                    eval_model(EH_model)
                    EH_model.train()        
    if pc.do_eval:
        eval_model(EH_model)  