#1/bin/bash

# CLASSPATH=$CLASSPATH:/home/SimCSE-main/LDCCorpus/preproc/ollie-app-latest.jar
CLASSPATH=$CLASSPATH:./ollie-app-latest.jar
CLASSPATH=$CLASSPATH:.

# INPUT_BASE=/home/SimCSE-main/LDCCorpus/gigaword_eng_5/data/nyt_eng_parse_2
INPUT_BASE=../gigaword_eng_5/data/nyt_eng_parse_2
# OUTPUT_BASE=/home/SimCSE-main/LDCCorpus/gigaword_eng_5/data/nyt_ollie_3
OUTPUT_BASE=../gigaword_eng_5/data/nyt_ollie_3

export JAVA_OPTS="-Xmx10g"

# compile OpenExtract.scala
scalac -classpath $CLASSPATH OpenExtract.scala

for year in $@; do
    scala -classpath $CLASSPATH ollie.OpenExtract $INPUT_BASE/$year $OUTPUT_BASE/$year.txt
done
# for year in $INPUT_BASE; do
#     scala -classpath $CLASSPATH ollie.OpenExtract $INPUT_BASE/$year $OUTPUT_BASE/$year.txt
# done


#nohup sh nyt_extract_event_3.sh 1995 1996 1997 1998 1999 2000 2001 2002 2003 2004 2005 2006 > extract_event.log 2>&1 &
#nohup sh nyt_extract_event_3.sh 2007 2008 2009 2010 > extract_event_2007_2010.log 2>&1 &