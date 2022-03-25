import torch.nn as nn
# def load_hard_similarity_dataset(path = '../event/resource/hard.txt'):
#     x_A = []
#     x_B = []
#     y = []
#     for line in open(path): 
#         pos_event1 = line.strip('\n').split('|')[0].strip(' ') + ' ' + line.strip('\n').split('|')[1].strip(' ') + ' ' + line.strip('\n').split('|')[2].strip(' ')
#         pos_event2 = line.strip('\n').split('|')[3].strip(' ') + ' ' + line.strip('\n').split('|')[4].strip(' ') + ' ' + line.strip('\n').split('|')[5].strip(' ')
#         x_A.append(pos_event1)
#         x_B.append(pos_event2)
#         y.append(1)
#         neg_event1 = line.strip('\n').split('|')[6].strip(' ') + ' ' + line.strip('\n').split('|')[7].strip(' ') + ' ' + line.strip('\n').split('|')[9].strip(' ')
#         neg_event2 = line.strip('\n').split('|')[9].strip(' ') + ' ' + line.strip('\n').split('|')[10].strip(' ') + ' ' + line.strip('\n').split('|')[11].strip(' ')
#         x_A.append(neg_event1)
#         x_B.append(neg_event2)
#         y.append(0)
#     return (x_A, x_B, y)
# load_hard_similarity_dataset()
class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp = 0.05):
        super().__init__()
        self.temp = temp #"Temperature for softmax."
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


def Event_Glove_CL_load_hard_similarity_dataset(path = './resource/hard.txt'):
    x_A = []
    x_B = []
    x_C = []
    x_D = []
    y = []
    for line in open(path): 
        pos_event1 = line.strip('\n').split('|')[0].strip(' ') + '|' + line.strip('\n').split('|')[1].strip(' ') + '|' + line.strip('\n').split('|')[2].strip(' ')
        pos_event2 = line.strip('\n').split('|')[3].strip(' ') + '|' + line.strip('\n').split('|')[4].strip(' ') + '|' + line.strip('\n').split('|')[5].strip(' ')
        x_A.append(pos_event1)
        x_B.append(pos_event2)
        neg_event1 = line.strip('\n').split('|')[6].strip(' ') + '|' + line.strip('\n').split('|')[7].strip(' ') + '|' + line.strip('\n').split('|')[8].strip(' ')
        neg_event2 = line.strip('\n').split('|')[9].strip(' ') + '|' + line.strip('\n').split('|')[10].strip(' ') + '|' + line.strip('\n').split('|')[11].strip(' ')
        x_C.append(neg_event1)
        x_D.append(neg_event2)
    return (x_A, x_B, x_C, x_D)



def TransitiveSentenceSimilarityDataset(path = "./resource/transitive.txt"):
    
    x_A = []
    x_B = []
    scores = []
    for line in open(path): 
        event1 = line.strip('\n').split('|')[0].strip(' ') + '|' + line.strip('\n').split('|')[1].strip(' ') + '|' + line.strip('\n').split('|')[2].strip(' ')
        event2 = line.strip('\n').split('|')[3].strip(' ') + '|' + line.strip('\n').split('|')[4].strip(' ') + '|' + line.strip('\n').split('|')[5].strip(' ')
        score = line.strip('\n').split('|')[6].strip(' ')
        x_A.append(event1)
        x_B.append(event2)
        scores.append(score)
    return (x_A, x_B, scores)
