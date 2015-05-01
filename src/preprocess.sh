#!/bin/bash

CSV_FILES=csv_files.txt
find ../data/pieces -type f -name '*.csv' > $CSV_FILES

NUM_LINES=$(wc -l < "$CSV_FILES")
for x in `seq 0 $(($NUM_LINES-1))`; do
  pypy preprocess_data.py $x &
done
wait

find ../data/pieces -name 'train_*.pkl' > train_files.txt
find ../data/pieces -name 'val_*.pkl' > val_files.txt
find ../data/pieces -name 'test_*.pkl' > test_files.txt

python merge_data.py train_files.txt val_files.txt test_files.txt
