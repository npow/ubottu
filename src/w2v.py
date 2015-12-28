import argparse
import cPickle
import csv
import gensim, logging
import nltk
import numpy as np
import xml.etree.ElementTree
from bs4 import BeautifulSoup

def str2bool(v):
      return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser()
parser.register('type','bool',str2bool)
parser.add_argument('--fname', type=str, default='trainset.csv', help='Input file name')
parser.add_argument('--run_w2v', type='bool', default=True, help='Run word2vec')
parser.add_argument('--dump_W', type='bool', default=True, help='Dump embeddings')
parser.add_argument('--window_size', type=int, default=7, help='Window size')
parser.add_argument('--embedding_size', type=int, default=300, help='Embedding size')
parser.add_argument('--min_count', type=int, default=1, help='Min count')
parser.add_argument('--num_workers', type=int, default=16, help='Num workers')
parser.add_argument('--split_utterances', type='bool', default=True, help='Split utterances')
parser.add_argument('--process_stackexchange', type='bool', default=True, help='Include stackexchange files')
parser.add_argument('--stackexchange_dir', type=str, default='.', help='Directory containing stackexchange files')
args = parser.parse_args()
print 'args: ', args

def get_stackexchange_lines(fname, elem):
    lines = []
    e = xml.etree.ElementTree.parse(fname).getroot()
    for row in e.findall('row'):
        body = row.get(elem)
        soup = BeautifulSoup(body)
        text = soup.get_text()
        for l in nltk.sent_tokenize(text):
            lines.append(l.replace('\n', ''))
    return lines

def get_lines():
    with open(args.fname, 'rb') as f:
        reader = csv.reader(f)
        lines = []
        for row in reader:
            c, r = row[0], row[1]
            if args.split_utterances:
                utterances = c.split('__EOS__') + [r]
            else:
                utterances = [c + ' ' + r]
            for utterance in utterances:
                tokens = utterance.split()
                lines.append(tokens)
        return lines

if args.run_w2v:
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    print 'loading lines...'
    lines = get_lines()
    if args.process_stackexchange:
        for d in ['meta.askubuntu.com', 'askubuntu.com']:
            for fname, elem in [('Posts.xml', 'Body'), ('meta.askubuntu.com/Comments.xml', 'Text')]:
                fname = '%s/%s/%s' % (args.stackexchange_dir, d, fname)
                se_lines = get_stackexchange_lines(fname, elem)
                print fname, len(se_lines)
                lines += se_lines

    print 'done loading lines: ', len(lines)
    model = gensim.models.Word2Vec(size=args.embedding_size, window=args.window_size, min_count=args.min_count, workers=args.num_workers)
    model.build_vocab(lines)
    model.train(lines)
    cPickle.dump(model, open('w2v_model_ws%s_d%d.pkl' % (args.window_size, args.embedding_size), 'wb'), protocol=-1)

if args.dump_W:
    model = cPickle.load(open('w2v_model_ws%s_d%s.pkl' % (args.window_size, args.embedding_size)))
    orig_W, word_idx_map = cPickle.load(open('dataset_1000000_updated/W.pkl'))
    W = np.random.uniform(-0.25, 0.25, (orig_W.shape[0], args.embedding_size)).astype(np.float32)
    num_skipped = 0
    for word in sorted(word_idx_map, key=word_idx_map.get, reverse=False):
        idx = word_idx_map[word]
        if not word in model:
            num_skipped += 1
            continue
        embedding = model[word]
        W[idx, :] = embedding
    cPickle.dump([W, word_idx_map], open('custom_ws%s_d%s_W.pkl' % (args.window_size, args.embedding_size), 'wb'), protocol=-1)
    print 'skipped: ', num_skipped, 'total: ', len(word_idx_map)
