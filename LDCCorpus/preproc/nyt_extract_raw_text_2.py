import xml.dom.minidom
import re
import os
import sys
import glob
import shutil  
import logging
import pathlib

# logging.basicConfig(level=logging.INFO, filename='raw_data_org.log' , filemode = 'w')


def extract_one_doc(filename):
    whitespace_pattern = re.compile(r'\s+') #匹配任意空白字符，等价于 [ \t\n\r\f]。
    try:
        
        dom = xml.dom.minidom.parse(filename)
        nitf = dom.documentElement
        blocks = nitf.getElementsByTagName('TEXT')
        text = []
        if len(blocks) != 1:
            print("{} 's text block is more than 1.".format(filename))
            return
        else: 
            for block in blocks[0].childNodes:
                try:
                    line = block.childNodes[0].data
                    line = whitespace_pattern.sub(' ', line)
                    text.append(line.strip(" "))
                except:
                    continue
        return text
    except Exception as e:
        print('Doc ', filename, "encounter the following erro: ", Exception)
        return None

    


if __name__ == '__main__':
#     # --------------------------------------------------把nyt corpus组织成对应的形式--------------------------------------------------
#     shutil.rmtree('/home/SimCSE-main/LDCCorpus/gigaword_eng_5/data/nyt_eng_parse')   
#     os.mkdir('/home/SimCSE-main/LDCCorpus/gigaword_eng_5/data/nyt_eng_parse')

#     input_dir = '/home/SimCSE-main/LDCCorpus/gigaword_eng_5/data/nyt_eng'
#     month_docs = glob.glob(os.path.join(input_dir, '*'))
#     month_docs = sorted(month_docs)
#     for month_doc in month_docs:
#         os.mkdir(os.path.join('/home/SimCSE-main/LDCCorpus/gigaword_eng_5/data/nyt_eng_parse', month_doc.split('_')[-1]))
#         with open(month_doc) as f:
#             lines = f.readlines()
#             for line in lines:
#                 # start_pattern = re.compile(r'<DOC id="NYT_ENG_\d+.\d+" type="story" >')
#                 start_pattern = re.compile(r'<DOC')
#                 m = start_pattern.match(line)
#                 if m is not None:
#                     date_pattern = re.compile(r'\D+')
#                     date, article_id = re.split(date_pattern, line)[1], re.split(date_pattern, line)[2]
#                     if not os.path.exists(os.path.join('/home/SimCSE-main/LDCCorpus/gigaword_eng_5/data/nyt_eng_parse', month_doc.split('_')[-1], date)):
#                         os.mkdir(os.path.join('/home/SimCSE-main/LDCCorpus/gigaword_eng_5/data/nyt_eng_parse', month_doc.split('_')[-1], date))       
#                     f2 = open(os.path.join('/home/SimCSE-main/LDCCorpus/gigaword_eng_5/data/nyt_eng_parse', month_doc.split('_')[-1], date, article_id + '.xml'), "w")
#                     logging.info('start writing', os.path.join('/home/SimCSE-main/LDCCorpus/gigaword_eng_5/data/nyt_eng_parse', month_doc.split('_')[-1], date, article_id), )
#                     f2.write(line)
#                 else:
#                     f2.write(line)
# # --------------------------------------------------把nyt corpus组织成对应的形式--------------------------------------------------



    # root_input_dir = '/home/SimCSE-main/LDCCorpus/gigaword_eng_5/data/nyt_eng_parse'
    root_input_dir = '../gigaword_eng_5/data/nyt_eng_parse'
    # output_dir = '/home/SimCSE-main/LDCCorpus/gigaword_eng_5/data/nyt_eng_parse_2'
    output_dir = '../gigaword_eng_5/data/nyt_eng_parse_2'
    for month_dir in os.listdir(root_input_dir):
        input_dir = os.path.join(root_input_dir, month_dir)
        day_dirs = glob.glob(os.path.join(input_dir, '*'))
        day_dirs = sorted(day_dirs)

        for day_dir in day_dirs:
            if os.path.isdir(day_dir):
                dir_name = os.path.relpath(day_dir, input_dir)
                # dir_name = os.path.join(output_dir, dir_name) ##原来的组织形式
                # os.makedirs(dir_name, exist_ok=True) ##原来的组织形式
                year, month, day = dir_name[0:4], dir_name[4:6], dir_name[6:]
                pathlib.Path(os.path.join(output_dir, year, year+month, year+month+day)).mkdir(parents=True, exist_ok=True) 

                docs = glob.glob(os.path.join(day_dir, '*'))
                docs = sorted(docs)
                for doc in docs:
                    if os.path.isfile(doc):
                        text = extract_one_doc(doc)
                        output_file = open(os.path.join(output_dir, year, year+month, year+month+day, year+month+day + '_' + os.path.basename(doc).replace('.xml', '.txt')), 'w')
                        # output_file = open(os.path.join(dir_name, os.path.basename(doc).replace('.xml', '.txt')), 'w')  ##原来的组织形式
                        if text is not None:
                            for line in text:
                                output_file.write(line + '\n')
                            output_file.close()

