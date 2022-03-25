#!/bin/bash

# BASE_DIR="/home/SimCSE-main/LDCCorpus/gigaword_eng_5/data/nyt_eng"
BASE_DIR="../gigaword_eng_5/data/nyt_eng"

for nyt_year_news in "$BASE_DIR"/*
do
    echo "$nyt_year_news"
    gunzip "$nyt_year_news"
done