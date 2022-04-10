
from utils import Similarity, Event_Glove_CL_load_hard_similarity_dataset, TransitiveSentenceSimilarityDataset
import torch.nn as nn
import torch
import logging
import scipy.stats
import numpy as np
def Hard_Similarity_eval(ECL_model):
    cosine_similarity = nn.CosineSimilarity(dim=1)
    input1, input2, input3, input4 = Event_Glove_CL_load_hard_similarity_dataset(path = './resource/hard.txt')
    num_correct = 0
    input1_id = torch.tensor(list(map(ECL_model.Glove.transform_eval, input1))).to(ECL_model.device)
    input2_id = torch.tensor(list(map(ECL_model.Glove.transform_eval, input2))).to(ECL_model.device)
    input3_id = torch.tensor(list(map(ECL_model.Glove.transform_eval, input3))).to(ECL_model.device)
    input4_id = torch.tensor(list(map(ECL_model.Glove.transform_eval, input4))).to(ECL_model.device)
    
    with torch.no_grad():
        input1 = ECL_model.pooler(ECL_model.composition_model.embeddings(input1_id), event_arg_embedding = None)
        input2 = ECL_model.pooler(ECL_model.composition_model.embeddings(input2_id), event_arg_embedding = None)
        input3 = ECL_model.pooler(ECL_model.composition_model.embeddings(input3_id), event_arg_embedding = None)
        input4 = ECL_model.pooler(ECL_model.composition_model.embeddings(input4_id), event_arg_embedding = None)
        pos_sim = cosine_similarity(input1, input2)
        neg_sim = cosine_similarity(input3, input4)
        
        num_correct = (pos_sim > neg_sim).sum().item()
        accuracy = num_correct / input1.shape[0]
        logging.info("Hard Similarity Task accurracy = {}".format(accuracy))
        
def Hard_Similarity_Extention_eval(ECL_model):
    cosine_similarity = nn.CosineSimilarity(dim=1)
    input1, input2, input3, input4 = Event_Glove_CL_load_hard_similarity_dataset(path = './resource/hard_extend.txt')
    num_correct = 0
    input1_id = torch.tensor(list(map(ECL_model.Glove.transform_eval, input1))).to(ECL_model.device)
    input2_id = torch.tensor(list(map(ECL_model.Glove.transform_eval, input2))).to(ECL_model.device)
    input3_id = torch.tensor(list(map(ECL_model.Glove.transform_eval, input3))).to(ECL_model.device)
    input4_id = torch.tensor(list(map(ECL_model.Glove.transform_eval, input4))).to(ECL_model.device)
    
    with torch.no_grad():
        input1 = ECL_model.pooler(ECL_model.composition_model.embeddings(input1_id), event_arg_embedding = None)
        input2 = ECL_model.pooler(ECL_model.composition_model.embeddings(input2_id), event_arg_embedding = None)
        input3 = ECL_model.pooler(ECL_model.composition_model.embeddings(input3_id), event_arg_embedding = None)
        input4 = ECL_model.pooler(ECL_model.composition_model.embeddings(input4_id), event_arg_embedding = None)
        pos_sim = cosine_similarity(input1, input2)
        neg_sim = cosine_similarity(input3, input4)
        
        num_correct = (pos_sim > neg_sim).sum().item()
        accuracy = num_correct / input1.shape[0]
        logging.info("Hard Similarity Extention Task accurracy = {}".format(accuracy))
        

       

def Transitive_eval(ECL_model):
    
    cosine_similarity = nn.CosineSimilarity(dim=1)
    event1, event2, scores = TransitiveSentenceSimilarityDataset(path = "./resource/transitive.txt") 
    event1_id = torch.tensor(list(map(ECL_model.Glove.transform_eval, event1))).to(ECL_model.device)
    event2_id = torch.tensor(list(map(ECL_model.Glove.transform_eval, event2))).to(ECL_model.device)
    with torch.no_grad():
        input1 = ECL_model.pooler(ECL_model.composition_model.embeddings(event1_id), event_arg_embedding = None)
        input2 = ECL_model.pooler(ECL_model.composition_model.embeddings(event2_id), event_arg_embedding = None)
        
        pred = cosine_similarity(input1, input2).cpu()
    
        pred = pred.detach().numpy()
        scores = np.array(scores)
        spearman_correlation, spearman_p = scipy.stats.spearmanr(pred, scores)
   
        # output_file = open("./log_Event_Glove_CL/Spearman/Spearman.log", 'w')
        # for score in pred:
        #     output_file.write(str(score) + '\n')
        # output_file.close()
        # logging.info('Output saved to ' + "./log_Event_Glove_CL/Spearman/Spearman.log")
        logging.info('Spearman correlation: ' + str(spearman_correlation))

    return 



def Hard_Similarity_only_hyper_eval(EH_model):
    cosine_similarity = nn.CosineSimilarity(dim=1)
    input1, input2, input3, input4 = Event_Glove_CL_load_hard_similarity_dataset(path = './resource/hard.txt', if_hyper=True)
    num_correct = 0


    
    with torch.no_grad():

        if EH_model.HyperGraph_Model.if_Related_Hyper:
            pass
        else:
            input1 = EH_model.HyperGraph_Model(EH_model.HyperGraph_Model.HyperGraphConstruction(input1))
            input2 = EH_model.HyperGraph_Model(EH_model.HyperGraph_Model.HyperGraphConstruction(input2))
            input3 = EH_model.HyperGraph_Model(EH_model.HyperGraph_Model.HyperGraphConstruction(input3))
            input4 = EH_model.HyperGraph_Model(EH_model.HyperGraph_Model.HyperGraphConstruction(input4))
        pos_sim = cosine_similarity(input1, input2)
        neg_sim = cosine_similarity(input3, input4)
        
        num_correct = (pos_sim > neg_sim).sum().item()
        accuracy = num_correct / input1.shape[0]
        logging.info("Hard Similarity Task accurracy = {}".format(accuracy))