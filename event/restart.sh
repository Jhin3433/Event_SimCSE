#!/bin/sh
ps -ef | grep python | cut -c 9-15| xargs kill -s 9

#前面几个短的 dropout是0.1 ，hg_initial_feature是100
#only_hyper
python Event_Glove_CL_onlyHyper.py -lr 2 -batch 64 -iav 0.01 -do_train -do_eval -hg_hidden_size 100 -hg_dropout 0.1 -hg_initial_feature 100 & 
python Event_Glove_CL_onlyHyper.py -lr 5 -batch 64 -iav 0.01 -do_train -do_eval -hg_hidden_size 100 -hg_dropout 0.1 -hg_initial_feature 100
