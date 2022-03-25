import os
import glob
import pathlib
# base_dir = "/home/SimCSE-main/LDCCorpus/gigaword_eng_5/data/nyt_eng_parse_2"
base_dir = "../gigaword_eng_5/data/nyt_eng_parse_2"
# output_dir = "/home/SimCSE-main/LDCCorpus/gigaword_eng_5/data/nyt_eng_parse_2+"
output_dir = "../gigaword_eng_5/data/nyt_eng_parse_2+"
# file_dir = "19940701_0001.txt"
# output_file = open("19940701_0001_reshape.txt", 'w')

#nyt_eng_parse_2中有的文件一行有多条句子，可能会影响ollie抽取结果。此代码用来将一行的多条句子 通过标点 拆分成一行只有一个句子
#有瑕疵，不予使用

for year in os.listdir(base_dir):
    for month in os.listdir(os.path.join(base_dir, year)):
        for day in os.listdir(os.path.join(base_dir, year, month)):
            pathlib.Path(os.path.join(output_dir, year, month, day)).mkdir(parents=True, exist_ok=True) 
            for doc in os.listdir(os.path.join(base_dir, year, month, day)):
                doc_path = os.path.join(base_dir, year, month, day, doc)
                if os.path.isfile(doc_path):
                    with open(doc_path) as f:
                        output_file = open(os.path.join(output_dir, year, month, day, doc), "w")
                        for line in f:
                            senteces = line.split(". ")
                            for sent in senteces:
                                if sent != senteces[-1]:
                                    output_file.writelines(sent + ".")
                                else:
                                    output_file.writelines(sent)
                        output_file.close()
                        f.close()



