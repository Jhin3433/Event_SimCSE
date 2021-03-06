import os
from torch.utils.data import Dataset

class Event_CL_nyt_dataset_6(Dataset):
    def __init__(self, Event_CL_dir):
        self.Event_Triples = []
        for path, dir_list, file_list in os.walk(Event_CL_dir):  
            for file_name in file_list:  
                with open(os.path.join(path, file_name), "r") as f:
                    for line in f:
                        if len(line.strip("\n").split(",")) == 3:
                            raw_event, positive_event, negtive_event = line.strip("\n").split(",")
                            continue
                        sub_raw_event, verb_raw_event, obj_raw_event = raw_event.strip().split("|")
                        sub_positive_event, verb_positive_event, obj_positive_event = positive_event.strip().split("|")
                        sub_negtive_event, verb_negtive_event, obj_negtive_event = negtive_event.strip().split("|")
                        # self.Event_Triples.append([SimpleTuple(sub_raw_event, verb_raw_event, obj_raw_event), SimpleTuple(sub_positive_event, verb_positive_event, obj_positive_event), SimpleTuple(sub_negtive_event, verb_negtive_event, obj_negtive_event)])
                        #只取只有单个词的
                        if len(sub_raw_event.split(" ")) ==  len(verb_raw_event.split(" ")) == len(obj_raw_event.split(" ")) == len(sub_positive_event.split(" ")) == len(verb_positive_event.split(" ")) == len(obj_positive_event.split(" ")) == len(sub_negtive_event.split(" ")) ==len(verb_negtive_event.split(" ")) ==len(obj_negtive_event.split(" ")) != 2:
                            self.Event_Triples.append([raw_event.strip(), positive_event.strip(), negtive_event.strip()])
                                            
    def __len__(self):
        return len(self.Event_Triples)

    def __getitem__(self, idx):
        
        Event_Triple = self.Event_Triples[idx]
        
        return Event_Triple[0], Event_Triple[1], Event_Triple[2]
        
        


class Event_CL_Dataset(Dataset):
    def __init__(self, Event_CL_file, Hyper_file  = None):

        self.Event_Triples = []
        with open(Event_CL_file, "r") as f:
            for line in f:
        
                split_line = line.split("||")
                synset_node = split_line[0].strip(" ")
                # pos_1_event = split_line[1].strip(" ")
                # pos_2_event = split_line[2].strip(" ")
                # neg_3_event = split_line[3].strip("\n").strip(" ")
                # self.Event_Triples.append([synset_node, pos_1_event.lower(), pos_2_event.lower(), neg_3_event.lower()])

    
                # pos_1_event, pos_1_arg = [x.lower() for x in split_line[1].strip(" ").split("<>")]
                pos_1_event, pos_1_arg = split_line[1].strip(" ").split("<>")
                pos_2_event, pos_2_arg = split_line[2].strip(" ").split("<>")
                neg_3_event, neg_3_arg = split_line[3].strip("\n").strip(" ").split("<>")
                self.Event_Triples.append([synset_node, (pos_1_event, pos_1_arg), (pos_2_event, pos_2_arg), (neg_3_event, neg_3_arg.split(",")[0])])
                #"都弄为小写, neg_event有多个arg时只需第一个"
                
              
                # if len(self.Event_Triples) > 1000 :
                #     break
        self.event_hyper = []
        if Hyper_file:
            with open(Event_CL_file, "r") as f:
                for line in f:
                    split_line = line.split("||")
                    self.event_hyper.append([split_line[0].strip(" "), split_line[1].strip(" "), split_line[2].strip("\n").strip(" ")])


    def __len__(self):
        return len(self.Event_Triples)
                
    def __getitem__(self, idx):
        
        Event_Triple = self.Event_Triples[idx]
        if self.event_hyper == []:
            return Event_Triple[0], Event_Triple[1], Event_Triple[2] ,Event_Triple[3]
        else:
            Event_Hyper = self.event_hyper[idx]
            return Event_Triple[0], Event_Triple[1], Event_Triple[2] ,Event_Triple[3], Event_Hyper[0], Event_Hyper[1], Event_Hyper[2]   