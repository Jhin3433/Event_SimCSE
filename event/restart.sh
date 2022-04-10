#!/bin/sh
ps -ef | grep python | cut -c 9-15| xargs kill -s 9
# python Event_Glove_CL_1.py -lr 5e-6 -batch 16 -iav 0.01 & python Event_Glove_CL_1.py -lr 5e-6 -batch 32 -iav 0.01 & python Event_Glove_CL_1.py -lr 5e-6 -batch 64 -iav 0.01
python Event_Glove_CL_1.py -lr 3e-5 -batch 16 -iav 0.01 & python Event_Glove_CL_1.py -lr 3e-5 -batch 32 -iav 0.01 & python Event_Glove_CL_1.py -lr 3e-5 -batch 64 -iav 0.01




# python Event_Glove_CL_1.py -lr 5e-6 -batch 64 & python Event_Glove_CL_1.py -lr 5e-6 -batch 128 & python Event_Glove_CL_1.py -lr 0.01 -batch 64 & python Event_Glove_CL_1.py -lr 0.01 -batch 128