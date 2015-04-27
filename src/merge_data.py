import cPickle
import gzip
import numpy as np
import random
import sys
from collections import Counter

FILE_LIST = sys.argv[1]
TRAIN_FILES = [f.strip() for f in open(FILE_LIST)]
VAL_FILE = '../data/valset.csv.pkl'
TEST_FILE = '../data/testset.csv.pkl'

W2V_FILE = '../embeddings/word2vec/GoogleNews-vectors-negative300.bin'
GLOVE_FILE = '../embeddings/glove/glove.840B.300d.txt'

def uniform_sample(a, b, k=0):
    if k == 0:
        return random.uniform(a, b)
    ret = np.zeros((k,))
    for x in xrange(k):
        ret[x] = random.uniform(a, b)
    return ret

def get_W(word_vecs, k):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k))            
    W[0] = np.zeros(k)
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word).lower()
                    break
                if ch != '\n':
                    word.append(ch)   
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
            else:
                f.read(binary_len)
    return word_vecs

def load_glove_vec(fname, vocab):
    """
    Loads word vecs from gloVe
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        for i,line in enumerate(f):
            L = line.split()
            word = L[0].lower()
            if word in vocab:
                word_vecs[word] = np.array(L[1:], dtype='float32')
    return word_vecs

def add_unknown_words(word_vecs, vocab, min_df=1, k=300, unk_token='**unknown**'):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = uniform_sample(-0.25,0.25,k)  
    word_vecs[unk_token] = uniform_sample(-0.25,0.25,k)

train_data = { 'c': [], 'r': [], 'y': [] }
train_vocab = Counter()
for pkl_file in TRAIN_FILES:
    cur_data, cur_vocab = cPickle.load(open(pkl_file))
    for key in cur_data:
        train_data[key] += cur_data[key]
    train_vocab += cur_vocab
    del cur_data, cur_vocab

val_data, val_vocab = cPickle.load(open(VAL_FILE))
test_data, test_vocab = cPickle.load(open(TEST_FILE))

vocab = train_vocab + val_vocab + test_vocab
del train_vocab, val_vocab, test_vocab

print "data loaded!"
print "num train: ", len(train_data['y'])
print "num val: ", len(val_data['y'])
print "num test: ", len(test_data['y'])
print "vocab size: ", len(vocab)

cPickle.dump([train_data, val_data, test_data], open('dataset.pkl', 'wb'))
del train_data, val_data, test_data

print "loading embeddings..."
#embeddings = load_bin_vec(W2V_FILE, vocab)
embeddings = load_glove_vec(GLOVE_FILE, vocab)

print "embeddings loaded!"
print "num words with embeddings: ", len(embeddings)

add_unknown_words(embeddings, vocab, min_df=10)
W, word_idx_map = get_W(embeddings, k=300)
print "W: ", W.shape
cPickle.dump([W, word_idx_map], open("W.pkl", "wb"))
del W

rand_vecs = {}
add_unknown_words(rand_vecs, vocab, min_df=10)
W2, _ = get_W(rand_vecs, k=300)
print "W2: ", W2.shape
cPickle.dump([W2, word_idx_map], open("W2.pkl", "wb"))
del W2

cPickle.dump(vocab, open('vocab.pkl', 'wb'))
del vocab

print "dataset created!"
