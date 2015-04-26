import cPickle
import csv
import gzip
import numpy as np
import os
import random
import re
import sys
from collections import defaultdict
from twokenize import tokenize

TRAIN_FILE = '../data/trainset_shuf.csv'
VAL_FILE = '../data/valset.csv'
TEST_FILE = '../data/testset.csv'

W2V_FILE = '../embeddings/word2vec/GoogleNews-vectors-negative300.bin'
GLOVE_FILE = '../embeddings/glove/glove.840B.300d.txt'

"""
os.environ['CLASSPATH']='.:../libs/commons-lang3-3.4.jar:../libs'
from jnius import autoclass
Twokenizer = autoclass('cmu.arktweetnlp.Twokenize')
def tokenize(s, tokenizer=Twokenizer()):
    s = s.replace('</s>', '')
    tokens = tokenizer.tokenizeRawTweetText(s.decode('utf-8'))
    return [tokens.get(i) for i in xrange(tokens.size())]
"""

def uniform_sample(a, b, k=0):
    if k == 0:
        return random.uniform(a, b)
    ret = np.zeros((k,))
    for x in xrange(k):
        ret[x] = random.uniform(a, b)
    return ret

def process_file(fname, vocab, clean_string=True):
    res = { 'c': [], 'r': [], 'y': [], 'wc_c': [], 'wc_r': []}
    for i,line in enumerate(csv.DictReader(open(fname))):
        if i % 10000 == 0:
            print i
        context, response, label = line['Context'], line['Response'], line['Correct']
        tok_context = process_line(context, clean_string)
        tok_response = process_line(response, clean_string)
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
#        res['wc_c'].append(len(words_c))
#        res['wc_r'].append(len(words_c))
    return res
    
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
                    word = ''.join(word)
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
            word = L[0]
            if word in vocab:
                word_vecs[word] = np.array(L[1:], dtype='float32')
    return word_vecs

def add_unknown_words(word_vecs, vocab, min_df=1, k=300, unk_token='**UNKNOWN**'):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = uniform_sample(-0.25,0.25,k)  
    word_vecs[unk_token] = uniform_sample(-0.25,0.25,k)
    
def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    #string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'m", " \'m", string) 
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r"`", " ` ", string)
    string = re.sub(r",", " , ", string) 
    #string = re.sub(r"!", " ! ", string) 
    #string = re.sub(r"\(", " \( ", string) 
    #string = re.sub(r"\)", " \) ", string) 
    #string = re.sub(r"\?", " \? ", string) 
    #string = re.sub(r"\s{2,}", " ", string)    
    string = string.replace('</s>', '__EOS__')
    return string.strip() if TREC else string.strip().lower()

def is_url(s):
    return s.startswith('http://') or s.startswith('https://') or s.startswith('ftp://') or s.startswith('ftps://') or s.startswith('smb://')
    try:
        val(s)
        return True
    except ValidationError:
        return False
    except ImportError:
        return False
    
def is_number(s):
  try:
    float(s)
    return True
  except ValueError:
    return False    

def NER(word):
    if is_url(word):
        return "__URL__"
    elif is_number(word):
        return "__NUMBER__"
    elif os.path.isabs(word):
        return "__PATH__"
    return word

def process_line(s, clean_string=True):
    if clean_string:
        s = clean_str(s)
    tokens = tokenize(s)
    tokens = [NER(token) for token in tokens]
    return tokens

print "loading data..."
vocab = defaultdict(float)
train_data = process_file(TRAIN_FILE, vocab)
val_data = process_file(VAL_FILE, vocab)
test_data = process_file(TEST_FILE, vocab)

if False:
    max_l = np.max(train_data['wc_c'] +
                   train_data['wc_r'] +
                   val_data['wc_c'] +
                   val_data['wc_r'] +
                   test_data['wc_c'] +
                   test_data['wc_r'])
else:
    max_l = None

print "data loaded!"
print "num train: ", len(train_data['y'])
print "num val: ", len(val_data['y'])
print "num test: ", len(test_data['y'])
print "vocab size: ", len(vocab)
print "max sentence length:\n", max_l

print "loading embeddings..."
#embeddings = load_bin_vec(W2V_FILE, vocab)
embeddings = load_glove_vec(GLOVE_FILE, vocab)

print "embeddings loaded!"
print "num words with embeddings: ", len(embeddings)

rand_vecs = {}
add_unknown_words(rand_vecs, vocab, min_df=10)
W2, _ = get_W(rand_vecs, k=300)
print "W2: ", W2.shape

add_unknown_words(embeddings, vocab, min_df=10)
W, word_idx_map = get_W(embeddings, k=300)
print "W: ", W.shape

if False:
    for key in ['wc_c', 'wc_r']:
        del train_data[key]
        del val_data[key]
        del test_data[key]

cPickle.dump([train_data, val_data, test_data, W, W2, word_idx_map, vocab], open("data.pkl", "wb"))
print "dataset created!"

