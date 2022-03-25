################################################
#       Just a script to convert to where each
#       document is on a seperate line
################################################
import sys
import os
def doc_on_line(filename, outfilename):
    infile = open(filename, 'r', encoding='utf-8')
    outfile = open(outfilename, 'w', encoding='utf-8')

    sentfill = "|SENT|" #seperator between different sentences in same doc
    tupfill = "|TUP|"  #seperator between tuples in the same sentence in the same doc

    prev_docid = "" 
    prev_sentid = ""
    docline = ""
    count = 0
    for line in infile:
        count +=1
        splits = line.split("|")
        docid=splits[0]
        sentid=splits[1]
        svo= "|".join(splits[2:5]) 
        sent=splits[5].strip()
        if docid == prev_docid: #still on the same document
            if sentid == prev_sentid: #still on the same sentence, same doc
                docline += tupfill + svo
            else: #same doc, new sentence
                docline +=  sentfill + docid + "|" + sentid + "|" + sent + tupfill + svo
                prev_sentid = sentid
        elif not prev_docid: #if we are on the first line
            prev_docid = docid
            prev_sentid = sentid
            docline =  docid + "|" + sentid + "|" + sent + tupfill + svo
        else: #on to a new document! print what we currently have
            prev_docid = docid
            prev_sentid = sentid
            outfile.write(docline + "\n")
            docline =  docid + "|" + sentid + "|" + sent + tupfill + svo
    print(filename , "has processed one line well.")
    infile.close()
    outfile.close()

if __name__ == "__main__":
    #method 1: 根据nyt_ollie_3中的所有文件来进行one_line提取
    # base_dir = "../gigaword_eng_5/data/nyt_ollie_3/"
    # output_dir = "../gigaword_eng_5/data/nyt_online_4/"
    # g = os.walk(base_dir, followlinks=True)  
    # for path,dir_list,file_list in g:  
    #     for file in file_list:
    #         infile = os.path.join(base_dir, file)
    #         outfile = os.path.join(output_dir, file)
    #         doc_on_line(infile, outfile)
    
    #method 2:根据命令行参数
    for argv in sys.argv:
        if ".py" in argv:
            continue
        infile = "../gigaword_eng_5/data/nyt_ollie_3/{}.txt".format(argv)
        outfile = "../gigaword_eng_5/data/nyt_online_4/{}.txt".format(argv)
        doc_on_line(infile, outfile)

# nohup python nyt_doc_on_line_4.py 1998 1999 2000 2001 2002 2003 2004 2005 2006 2007 2008 2009 2010 > nyt_doc_on_line_4.log 2>&1 &