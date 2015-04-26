import cPickle
import csv
import os
import re
import sys
from collections import Counter
from twokenize import tokenize

INPUT_FILE = sys.argv[1]
print INPUT_FILE

"""
os.environ['CLASSPATH']='.:../libs/commons-lang3-3.4.jar:../libs'
from jnius import autoclass
Twokenizer = autoclass('cmu.arktweetnlp.Twokenize')
def tokenize(s, tokenizer=Twokenizer()):
    s = s.replace('</s>', '')
    tokens = tokenizer.tokenizeRawTweetText(s.decode('utf-8'))
    return [tokens.get(i) for i in xrange(tokens.size())]
"""

def process_file(fname, vocab, clean_string=True):
    res = { 'c': [], 'r': [], 'y': [] }
    for i,line in enumerate(csv.reader(open(fname))):
        if i % 10000 == 0:
            print "%s: %d" % (fname, i)
        context, response, label = line[0], line[1], line[2]
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
    return res
      
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

vocab = Counter()
data = process_file(INPUT_FILE, vocab)
print INPUT_FILE, "vocab: ", len(vocab), " y: ", len(data['y'])
cPickle.dump([data, vocab], open('%s.pkl' % INPUT_FILE, 'wb'))
