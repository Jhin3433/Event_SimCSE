import random
import os

if __name__ == '__main__':
    # path = '/home/SimCSE-main/LDCCorpus/gigaword_eng_5'

    filenames = [

        '../gigaword_eng_5/data/nyt_ollie_3/1994.txt', ##未连接成one_line的svo , num_total = 2877182
        '../gigaword_eng_5/data/nyt_ollie_3/1995.txt', ##num_total= 9714779
        '../gigaword_eng_5/data/nyt_ollie_3/1996.txt', ##num_total= 12409883
        '../gigaword_eng_5/data/nyt_ollie_3/1997.txt', ##num_total= 12529448
        '../gigaword_eng_5/data/nyt_ollie_3/1998.txt', ##num_total= 9905589 
        '../gigaword_eng_5/data/nyt_ollie_3/1999.txt', ##num_total= 9847540
        '../gigaword_eng_5/data/nyt_ollie_3/2000.txt', ##num_total= 9176640 
        '../gigaword_eng_5/data/nyt_ollie_3/2001.txt', ##num_total= 10020313 
        '../gigaword_eng_5/data/nyt_ollie_3/2002.txt', ##num_total= 10633856
        '../gigaword_eng_5/data/nyt_ollie_3/2003.txt', ##num_total= 2017555 
        '../gigaword_eng_5/data/nyt_ollie_3/2004.txt', ##num_total= 5929855
        '../gigaword_eng_5/data/nyt_ollie_3/2005.txt', ##num_total= 10081781 
        '../gigaword_eng_5/data/nyt_ollie_3/2006.txt', ##num_total= 10244886 
        '../gigaword_eng_5/data/nyt_ollie_3/2007.txt', ##num_total= 8640610
        '../gigaword_eng_5/data/nyt_ollie_3/2008.txt', ##num_total= 7689898
        '../gigaword_eng_5/data/nyt_ollie_3/2009.txt', ##num_total= 6026841
        '../gigaword_eng_5/data/nyt_ollie_3/2010.txt', ##num_total= 6357120



 
    ]

    instances = []
    for filename in filenames:
        instances += open(filename, 'r').readlines()
    num_total = len(instances)

    def generate(num, output_file):
        indices = random.sample(range(num_total), num) #从num_total中随机挑选选取num---由num_dict定义
        samples = [instances[index] for index in indices]
        f = open(output_file, 'w')
        for line in samples:
            f.write(line)
        f.close()
        print(output_file + ' done')

    num_dict = {
        # 'data/nyt_final/1987_neg.txt': 6423165,
        # 'data/nyt_final/1988_neg.txt': 6491698,
        # 'data/nyt_final/1989_neg.txt': 6347525,
        # 'data/nyt_final/1990_neg.txt': 6243159,
        # 'data/nyt_final/1991_neg.txt': 5559770,
        # 'data/nyt_final/1992_neg.txt': 5447308,
        # 'data/nyt_final/1993_neg.txt': 5324032,


        # 'data/nyt_final/1994_neg.txt': 5288859,
        '../gigaword_eng_5/data/nyt_neg_5/1994_neg.txt': 2877182,
        '../gigaword_eng_5/data/nyt_neg_5/1995_neg.txt': 9714779,
        '../gigaword_eng_5/data/nyt_neg_5/1996_neg.txt': 12409883,
        '../gigaword_eng_5/data/nyt_neg_5/1997_neg.txt': 12529448,
        '../gigaword_eng_5/data/nyt_neg_5/1998_neg.txt': 9905589,
        '../gigaword_eng_5/data/nyt_neg_5/1999_neg.txt': 9847540,
        '../gigaword_eng_5/data/nyt_neg_5/2000_neg.txt': 9176640,
        '../gigaword_eng_5/data/nyt_neg_5/2001_neg.txt': 10020313,
        '../gigaword_eng_5/data/nyt_neg_5/2002_neg.txt': 10633856,
        '../gigaword_eng_5/data/nyt_neg_5/2003_neg.txt': 2017555,
        '../gigaword_eng_5/data/nyt_neg_5/2004_neg.txt': 5929855,
        '../gigaword_eng_5/data/nyt_neg_5/2005_neg.txt': 10081781,
        '../gigaword_eng_5/data/nyt_neg_5/2006_neg.txt': 10244886,
        '../gigaword_eng_5/data/nyt_neg_5/2007_neg.txt': 8640610,
        '../gigaword_eng_5/data/nyt_neg_5/2008_neg.txt': 7689898,
        '../gigaword_eng_5/data/nyt_neg_5/2009_neg.txt': 6026841,
        '../gigaword_eng_5/data/nyt_neg_5/2010_neg.txt': 6357120,

        # 'data/nyt_final/1995_neg.txt': 5744247,
        # 'data/nyt_final/1996_neg.txt': 5772160,
        # 'data/nyt_final/1997_neg.txt': 5808079,
        # 'data/nyt_final/1998_neg.txt': 6424286,
        # 'data/nyt_final/1999_neg.txt': 6594287,
        # 'data/nyt_final/2000_neg.txt': 6977080,
        # 'data/nyt_final/2001_neg.txt': 6864116,
        # 'data/nyt_final/2002_neg.txt': 6978644,
        # 'data/nyt_final/2003_neg.txt': 6837301,
        # 'data/nyt_final/2004_neg.txt': 6703959,
        # 'data/nyt_final/2005_neg.txt': 6614429,
        # 'data/nyt_final/2006_neg.txt': 6662519,
        # 'data/nyt_final/2007_neg.txt': 3106803,
    }

    for filename in num_dict:
        num = num_dict[filename]
        generate(num, filename)
