import json
import itertools
import copy
import pandas as pd



num_only_one_example_sense = 0
num_only_one_sense_verb = 0
def remove_pair_same_exmple(sense_examples):
    if sense_examples[0]!= sense_examples[1]:
        return sense_examples 

def positive_dataset_prepare(verb_sense_dict):
    pos_sentence_pair_dataset = {}
    for index, sense in enumerate(verb_sense_dict['senses']):
        if len(sense['examples']) > 1: #刨除只有一个example的sense    #############有1064个只有一个example的sense？？
            sense_examples = list(itertools.product(sense['examples'], sense['examples']))      
            filtered_sense_examples = list(filter(remove_pair_same_exmple, sense_examples)) #每个sense的exmaple两两组合
            pos_sentence_pair_dataset[verb_sense_dict['lemma'] + str(index) + '-pos'] = filtered_sense_examples #后续可以在此基础上作lemmatization, 只有成对sentence
        else:
            global num_only_one_example_sense
            num_only_one_example_sense += 1
            print("num_only_one_example_sense = {}".format(num_only_one_example_sense))
    return pos_sentence_pair_dataset


def remove_empty_neg(neg_sentence_pair_dataset):
    if neg_sentence_pair_dataset != {}:
        return neg_sentence_pair_dataset 
    else:
        global num_only_one_sense_verb
        num_only_one_sense_verb += 1
        print("num_only_one_sense_verb = {}".format(num_only_one_sense_verb))

def negative_dataset_prepare(verb_sense_dict):
    neg_sentence_pair_dataset = {}
    for index in list(itertools.product(list(range(len(verb_sense_dict['senses']))), list(range(len(verb_sense_dict['senses']))))):
        if index[0]!= index[1] and index[0] < index[1] :
            sense_examples = list(itertools.product(verb_sense_dict['senses'][index[0]]['examples'], verb_sense_dict['senses'][index[1]]['examples']))   
            neg_sentence_pair_dataset[verb_sense_dict['lemma'] + str(index[0]) + str(index[1])  + '-neg'] = sense_examples #每个sense的exmaple两两组合
    return neg_sentence_pair_dataset

def InvalidFunction():
    with open('./resource/verb_sense_dict.json','r',encoding='utf8')as fp:
        verb_sense_dict = json.load(fp)
    pos_sentence_pair_dataset = list(map(positive_dataset_prepare, verb_sense_dict))

    neg_sentence_pair_dataset = list(map(negative_dataset_prepare, verb_sense_dict))
    neg_sentence_pair_dataset = list(filter(remove_empty_neg, neg_sentence_pair_dataset)) ##刨除只有一个sense的verb    #############有523个只有一个sense的verb
    print("Function finished!")

def positive_negative_dataset_prepare(verb_sense_dict):
    positive_negative_dataset = []
    for index, sense in enumerate(verb_sense_dict['senses']):
        ##一个verb的一个sense，多个句子两两组合
        temp_positive_negative_dataset = []
        if len(sense['examples']) > 1: #刨除只有一个example的sense    #############有多少个只有一个example的sense
            pos_filtered_sense_examples = list(itertools.combinations(sense['examples'], r =2)) 
            for pos_filtered_sense_example in pos_filtered_sense_examples:
                ##添加prompt提示词
                pos_filtered_sense_example = list(pos_filtered_sense_example)
                pos_filtered_sense_example[0] += " The center word is {}.".format(verb_sense_dict['lemma'].split('-')[0])
                pos_filtered_sense_example[1] += " The center word is {}.".format(verb_sense_dict['lemma'].split('-')[0])

                temp_positive_negative_dataset.append({pos_filtered_sense_example[0]:verb_sense_dict['lemma'] + str(index), pos_filtered_sense_example[1]:verb_sense_dict['lemma'] + str(index)})

            ##negative sentence ready, 非当前sense的其他sense的所有句子
            neg_sentence = []
            for index2, sense2 in enumerate(verb_sense_dict['senses']):
                if index != index2:
                    for sentence_example in sense2['examples']:
                        ##添加prompt提示词
                        sentence_example += " The center word is {}.".format(verb_sense_dict['lemma'].split('-')[0])
                        neg_sentence.append({sentence_example:verb_sense_dict['lemma'] + str(index2)})

            ##加上negative sentence
            for temp in temp_positive_negative_dataset:
                temp_list = [] 
                for j in range(len(neg_sentence)):
                    temp_list.append(copy.deepcopy(temp))
                for i, temp2 in enumerate(temp_list):
                    for k, v in neg_sentence[i].items():
                        temp2[k] = v
                positive_negative_dataset.extend(temp_list)   
    return positive_negative_dataset
                    

def save_sentence_triple_csv(sentence_triple_dataset):
    sent0 = []
    sent1 = []
    hard_neg = []
    sent0_label = []
    sent1_label = []
    hard_neg_label = []
    for verb in sentence_triple_dataset:
        for sentence_triple in verb:
            if len(sentence_triple.items()) > 2: #arise 同一个example出现在了两个sense中
                sent0.append(list(sentence_triple.items())[0][0])
                sent0_label.append(list(sentence_triple.items())[0][1])

                sent1.append(list(sentence_triple.items())[1][0])
                sent1_label.append(list(sentence_triple.items())[1][1])       

                hard_neg.append(list(sentence_triple.items())[2][0])
                hard_neg_label.append(list(sentence_triple.items())[2][1])   

    df = pd.DataFrame({'sent0':sent0, 'sent1':sent1, 'hard_neg':hard_neg, 'sent0_label':sent0_label, 'sent1_label':sent1_label, 'hard_neg_label':hard_neg_label})
    df.to_csv('../data/verb_senses_for_simcse.csv', index = None)
    return

def main():
    with open('./resource/verb_sense_dict.json','r',encoding='utf8')as fp:
            verb_sense_dict = json.load(fp)
                
    sentence_triple_dataset = list(map(positive_negative_dataset_prepare, verb_sense_dict))
    filter_empty_sentence_triple_dataset = list(filter(lambda x : x != [], sentence_triple_dataset))
    save_sentence_triple_csv(filter_empty_sentence_triple_dataset)
    print(1)

if __name__ == "__main__":
    main()
    # InvalidFunction()