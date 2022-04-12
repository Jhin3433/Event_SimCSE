
import random

# 此文件的作用是将产生的三元组 数据集tirple_events.txt 顺序打乱，并将其拆分成小份


def preprocess_dataset(file_name = 'LMMS_Pickle/hard_verb_centric_triple_events.txt'):
    open_diff = open(file_name, 'r') # 源文本文件
    diff_line = open_diff.readlines()

    line_list = []
    for line in diff_line:
        line_list.append(line)

    random.shuffle(line_list)


    count = len(line_list) # 文件行数
    print('源文件数据行数：',count)
    # 切分diff
    diff_match_split = [line_list[i:i+2500000] for i in range(0,len(line_list),2500000)]# 每个文件的数据行数

    # 将切分的写入多个txt中
    for i,j in zip(range(0,int(count/2500000+1)),range(0,int(count/2500000+1))): # 写入txt，计算需要写入的文件数
        with open('./hard_triple_events/triple_events_%d.txt'% j,'w+') as temp:
            for line in diff_match_split[i]:
                temp.write(line)
    print('拆分后文件的个数：',i+1)


def shuffle_dataset(file_name = './no_other_hyper/hard_hyper_verb_centric_triple_events.txt'):
    open_diff = open(file_name, 'r') # 源文本文件
    diff_line = open_diff.readlines()

    line_list = []
    for line in diff_line:
        line_list.append(line)

    random.shuffle(line_list)


    count = len(line_list) # 文件行数
    print('源文件数据行数：',count)
    # 切分diff
    diff_match_split = [line_list[i:i+100000000] for i in range(0,len(line_list),100000000)]# 每个文件的数据行数

    # 将切分的写入多个txt中
    for i,j in zip(range(0,int(count/100000000+1)),range(0,int(count/100000000+1))): # 写入txt，计算需要写入的文件数
        with open('./hard_triple_events/hard_hyper_verb_centric_triple_events.txt','w+') as temp:
            for line in diff_match_split[i]:
                temp.write(line)
    print('拆分后文件的个数：',i+1)

def lemma_func(file_name):
    open_diff = open(file_name, 'r') # 源文本文件
    diff_line = open_diff.readlines()

    line_list = []
    for line in diff_line:
            split_line = line.split("||")
            synset_node = split_line[0].strip(" ")
            pos_1_event, pos_1_arg = split_line[1].strip(" ").split("<>")
            pos_2_event, pos_2_arg = split_line[2].strip(" ").split("<>")
            neg_3_event, neg_3_arg = split_line[3].strip("\n").strip(" ").split("<>")


if __name__ == "__main__":
    # preprocess_dataset()
    # shuffle_dataset(file_name = './no_other_hyper/hard_hyper_verb_centric_triple_events.txt')
    file_name = './no_other_hyper/hard_hyper_verb_centric_triple_events.txt'
    lemma_func(file_name)