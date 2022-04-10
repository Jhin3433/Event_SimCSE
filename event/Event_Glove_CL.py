import os
from numpy import float128, float32
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import logging
from Eval_func import Hard_Similarity_eval, Hard_Similarity_Extention_eval, Transitive_eval
import argparse

from Event_CL_Dataset import Event_CL_Dataset
import scipy.sparse as sp
from HyperGraph import target_event_Hypergraph_prepare
import sys
from Model import Event_CL


#此版本加入超图
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
 
        self.if_Related_Hyper = args.ifhyper
        # HyperGraph_Model Parameters
        self.hg_hidden_size = 100
        self.hg_dropout = 0.3
        self.hg_initial_feature = 100
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='The parameter of The model.')
    #type是要传入的参数的数据类型  help是该参数的提示信息
    parser.add_argument('-lr', type=float128, default=5e-6, help='learning rate')
    parser.add_argument('-batch', type=int, default=64, help='batch size')
    parser.add_argument('-iav', type=float32, default=0.1, help='initial_accumulator_value')
    parser.add_argument('-ifhyper', type=bool, default=False, help='if use events doc to construct hyper')



    arg_str = ''
    args = parser.parse_args()
    for arg in vars(args):
        arg_str = arg_str + "_" + arg + "_" + str(getattr(args, arg))
    
    
    logging.basicConfig(filename='./log_Event_Glove_CL/CL_Hyper/Event_Glove_CL{}.log'.format(arg_str), format='%(asctime)s | %(levelname)s | %(message)s', level=logging.DEBUG, filemode='w') #有filename是文件日志输出,filemode是’w’的话，文件会被覆盖之前生成的文件会被覆盖

    seed = 42
    torch.manual_seed(seed)
    pc = Parameter_Config(args)
    

    ECL_model = Event_CL(pc, pool_type = pc.pool_type)#memory:1.4G
    ECL_model.to(pc.device)
    # ECL_model.dict_event_hyper, ECL_model.arg_related_synset = target_event_Hypergraph_prepare(ECL_model)

    
    if pc.do_train:
        
        # Event_CL_dir = "../LDCCorpus/gigaword_eng_5/data/nyt_dataset_6" #对应Event_CL_nyt_dataset_6
        # LMMS_Pickle文件夹中包含全部的数据，经过shuffle_split.py处理后保存在hard_triple_events文件夹下
        Event_CL_file = "./resource/no_other_hyper/hard_hyper_verb_centric_triple_events.txt"
        if pc.if_Related_Hyper:
            Hyper_file = "./resource/hard_triple_events/hyper_events_0.txt"
            ECLD = Event_CL_Dataset(Event_CL_file, Hyper_file = Hyper_file)   
        else:  
            ECLD = Event_CL_Dataset(Event_CL_file) 
        ECL_model.HyperGraph_init(pc)#memory:1.6G
        
        
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

                if pc.if_Related_Hyper:
                    raw_event_hyper = event_data[4]
                    pos_event_hyper = event_data[5]
                    neg_event_hyper = event_data[6]

                else:
                    raw_event_hyper = event_data[1][0]
                    pos_event_hyper = event_data[2][0]
                    pos_event_hyper = event_data[3][0]

                raw_event_hypergraph = ECL_model.HyperGraph_Model(ECL_model.HyperGraph_Model.HyperGraphConstruction(raw_event_hyper))
                pos_event_hypergraph = ECL_model.HyperGraph_Model(ECL_model.HyperGraph_Model.HyperGraphConstruction(pos_event_hyper))
                neg_event_hypergraph = ECL_model.HyperGraph_Model(ECL_model.HyperGraph_Model.HyperGraphConstruction(neg_event_hyper))
                
                #comprehensive embeddings
                pooler_raw_event_embeddings = ECL_model.pooler(raw_event_embeddings, raw_event_arg_embeddings, raw_event_hypergraph, pool_type = pc.pool_type)
                pooler_pos_event_embeddings = ECL_model.pooler(pos_event_embeddings, pos_event_arg_embeddings, pos_event_hypergraph, pool_type = pc.pool_type)
                pooler_neg_event_embeddings = ECL_model.pooler(neg_event_embeddings, neg_event_arg_embeddings, neg_event_hypergraph, pool_type = pc.pool_type)
                
                
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