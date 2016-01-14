import cPickle
import csv
import os
import re
import sys
from collections import Counter

FILE_INDEX = int(sys.argv[1])
FILES = [f.strip() for f in open('csv_files.txt')]
INPUT_FILE = FILES[FILE_INDEX]
print INPUT_FILE

def process_file(fname, vocab, clean_string=True):
    res = { 'c': [], 'r': [], 'y': [] }
    for i,line in enumerate(csv.reader(open(fname))):
        if i % 10000 == 0:
            print "%s: %d" % (fname, i)
        assert(len(line) == 3)
        context, response, label = line[0], line[1], line[2]
        tok_context = context.split()
        tok_response = response.split()
        context = ' '.join(tok_context)
        response = ' '.join(tok_response)
        words_c = set(tok_context)
        words_r = set(tok_response)
        words = words_c | words_r
        for word in words:
            vocab[word] += 1
        res['c'].append(context)
        res['r'].append(response)
        res['y'].append(label)
    return res
      
vocab = Counter()
data = process_file(INPUT_FILE, vocab)
print INPUT_FILE, "vocab: ", len(vocab), " y: ", len(data['y'])
cPickle.dump([data, vocab], open('%s.pkl' % INPUT_FILE, 'wb'), protocol=-1)
