import sys
sys.path.insert(0, '.')
from event_tensors.train_utils import EventPredQueuedInstances
from event_tensors.glove_utils import Glove
import os

if __name__ == '__main__':


    svo_file_dir = "../gigaword_eng_5/data/nyt_online_4" #1994_oneline.txt  已连接成one_line
    neg_svo_file_dir = "../gigaword_eng_5/data/nyt_neg_5" #1994_neg.txt未连接成oneline
    output_file_dir = "../gigaword_eng_5/data/nyt_dataset_6"
    
    svo_files = []
    for parent, dirnames, filenames in os.walk(svo_file_dir,  followlinks=True):
        for filename in filenames:
            svo_file = os.path.join(parent, filename)
            svo_files.append(svo_file)
    neg_svo_files = []
    for parent, dirnames, filenames in os.walk(neg_svo_file_dir,  followlinks=True):
        for filename in filenames:
            neg_svo_file = os.path.join(parent, filename)
            neg_svo_files.append(neg_svo_file)
    svo_files.sort()
    neg_svo_files.sort()
    for svo_file, neg_svo_file in zip(svo_files, neg_svo_files):
        if svo_file.split('/')[-1][0:4] == neg_svo_file.split('/')[-1][0:4]:
            output_file = os.path.join(output_file_dir, svo_file.split('/')[-1][0:4] + '_dataset.txt')
        # svo_file = sys.argv[1]
        # neg_svo_file = sys.argv[2]
        # output_file = sys.argv[3]

        emb_file = 'data/glove.6B.100d.ext.txt'
        num_queues = 256
        batch_size = 128
        max_phrase_size = 10
        embeddings = Glove(emb_file)
        id2word = embeddings.reverse_dict()
        data = list(iter(EventPredQueuedInstances(svo_file, neg_svo_file, embeddings, num_queues, batch_size, max_phrase_size)))[:-1]  # remove None at the end
        # output_file = open('data/event_prediction_small.txt', 'w')
        output_file = open(output_file, 'w')
        for ei, et, en in data:
            ei_subj, ei_verb, ei_obj = ei
            ei_subj = [id2word[i] for i in ei_subj[0] if i != 1]
            ei_verb = [id2word[i] for i in ei_verb[0] if i != 1]
            ei_obj = [id2word[i] for i in ei_obj[0] if i != 1]
            et_subj, et_verb, et_obj = et
            et_subj = [id2word[i] for i in et_subj[0] if i != 1]
            et_verb = [id2word[i] for i in et_verb[0] if i != 1]
            et_obj = [id2word[i] for i in et_obj[0] if i != 1]
            en_subj, en_verb, en_obj = en
            en_subj = [id2word[i] for i in en_subj[0] if i != 1]
            en_verb = [id2word[i] for i in en_verb[0] if i != 1]
            en_obj = [id2word[i] for i in en_obj[0] if i != 1]
            if len(ei_subj) == 0:
                continue
            if len(ei_verb) == 0:
                continue
            if len(ei_obj) == 0:
                continue
            if len(et_subj) == 0:
                continue
            if len(et_verb) == 0:
                continue
            if len(et_obj) == 0:
                continue
            if len(en_subj) == 0:
                continue
            if len(en_verb) == 0:
                continue
            if len(en_obj) == 0:
                continue
            ei_subj = ' '.join(ei_subj)
            ei_verb = ' '.join(ei_verb)
            ei_obj = ' '.join(ei_obj)
            ei = '|'.join([ei_subj, ei_verb, ei_obj])
            et_subj = ' '.join(et_subj)
            et_verb = ' '.join(et_verb)
            et_obj = ' '.join(et_obj)
            et = '|'.join([et_subj, et_verb, et_obj])
            en_subj = ' '.join(en_subj)
            en_verb = ' '.join(en_verb)
            en_obj = ' '.join(en_obj)
            en = '|'.join([en_subj, en_verb, en_obj])
            line = ', '.join([ei, et, en]) + '\n'
            output_file.write(line)
        output_file.close()
