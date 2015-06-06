import cPickle
import csv
import os
import re
import sys
from collections import Counter

INPUT_FILE = sys.argv[1]
OUTPUT_FILE = sys.argv[2]
print INPUT_FILE, OUTPUT_FILE

def process_file(fname):
    res = { 'c': [], 'r': [], 'y': [] }
    with open('pv_sents.txt', 'wb') as f:
        sent2idx = {}
        for i,line in enumerate(csv.reader(open(fname))):
            if i % 10000 == 0:
                print "%s: %d" % (fname, i)
            context, response, label = line[0], line[1], int(line[2])

            context_sents = [s.strip() for s in context.split('</s>')]
            response_sents = [s.strip() for s in response.split('</s>')]

            for s in context_sents + response_sents:
                if not s in sent2idx:
                    sent2idx[s] = len(sent2idx)
                    f.write('%s\n' % s)

            context_idxs = [sent2idx[s] for s in context_sents]
            response_idxs = [sent2idx[s] for s in response_sents]
            res['c'].append(context_idxs)
            res['r'].append(response_idxs)
            res['y'].append(label)
        return res
      
data = process_file(INPUT_FILE)
cPickle.dump(data, open(OUTPUT_FILE, 'wb'), protocol=-1)
