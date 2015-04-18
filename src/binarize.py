import csv
import joblib
import numpy as np

TRAIN_FILE = 'data/trainset.csv'
VAL_FILE = 'data/valset.csv'
TEST_FILE = 'data/testset.csv'

UNK_TOKEN = '*UNKNOWN*'

def get_vocabulary(fname, vocab, wc):
    with open(fname, 'rb') as f:
        reader = csv.reader(f)
        for i,row in enumerate(reader):
            if i == 0:
                continue

            if i % 10000 == 0:
                print "get_vocabulary, fname: %s, line: %d" % (fname, i)
            context, response, label = row[:3]

            tok_context = tokenize(context)
            tok_response = tokenize(response)

            for w in tok_context + tok_response:
                if not w in vocab:
                    vocab[w] = len(vocab)
                    wc[w] = 0
                wc[w] += 1

def tokenize(s):
    s = s.replace('</s>', '')
    return s.split()
    s = ' '.join(s.split('</s>'))
    tokens = tokenizer.tokenizeRawTweetText(s.decode('utf-8'))
    return [tokens.get(i) for i in xrange(tokens.size())]

def binarize(s, vocab):
    tokens = tokenize(s)
    return np.array([vocab.get(word, vocab[UNK_TOKEN]) for word in tokens], dtype=np.int32).reshape((1,-1))

def binarize_file(fname, vocab):
    L = []
    with open(fname, 'rb') as f:
        reader = csv.reader(f)
        for i,row in enumerate(reader):
            if i == 0:
                continue
            print "binarize_file, fname: %s, line: %d" % (fname, i)
            context, response, label = row[:3]
            L.append([binarize(context, vocab), binarize(response, vocab), int(label)])
    joblib.dump(L, fname + '.rand.pkl')

def main():
    vocab = {}
    wc = {}
    get_vocabulary(TRAIN_FILE, vocab, wc)
    get_vocabulary(VAL_FILE, vocab, wc)
    get_vocabulary(TEST_FILE, vocab, wc)
#    vocab[UNK_TOKEN] = len(vocab)
    joblib.dump(vocab, 'vocab.pkl')
    joblib.dump(wc, 'wc.pkl')

#    binarize_file(VAL_FILE, vocab)
#    binarize_file(TEST_FILE, vocab)
#    binarize_file(TRAIN_FILE, vocab)

if __name__ == '__main__':
    main()
