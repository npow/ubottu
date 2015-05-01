#!/bin/bash

TRAIN_FILE=trainset.csv
VAL_FILE=valset.csv
TEST_FILE=testset.csv

split -a 4 -l 8000 $TRAIN_FILE splittrain_ 
split -a 4 -l 8000 $VAL_FILE splitval_ 
split -a 4 -l 8000 $TEST_FILE splittest_ 

rm -rf pieces
mkdir -p pieces/train pieces/val pieces/test

find . -name 'splittrain_*' | gawk '{ printf "mv %s pieces/train/train_%d.csv\n", $0, NR }' | bash
find . -name 'splitval_*' | gawk '{ printf "mv %s pieces/val/val_%d.csv\n", $0, NR }' | bash
find . -name 'splittest_*' | gawk '{ printf "mv %s pieces/test/test_%d.csv\n", $0, NR }' | bash

