#!/bin/bash

PREFIX=../data/splittrain_

rm -f ${PREFIX}*.pkl
for x in ${PREFIX}* ../data/valset.csv ../data/testset.csv; do
  python preprocess_data.py $x &
done
wait

ls ${PREFIX}*.pkl > files.txt
python merge_data.py files.txt
