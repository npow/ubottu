from __future__ import division
import cPickle
import itertools
import numpy as np

_, _, test_data = cPickle.load(open('dataset_ibm/blobs/dataset.pkl'))
test_probas = cPickle.load(open('test_probas.pkl'))
print test_probas.shape
W, word_idx_map = cPickle.load(open('dataset_ibm/blobs/W.pkl'))

inv_vocab = {}
for w in word_idx_map:
    inv_vocab[word_idx_map[w]] = w

def to_words(indices):
    return ' '.join([inv_vocab[i] for i in indices])

def generate_report(probas, k=5, group_size=10):
    test_size = 10
    n_batches = len(probas) // test_size
    n_correct = 0
    L_correct, L_incorrect = [], []
    html_correct = ''
    html_incorrect = ''
    for i in xrange(n_batches):
        batch = np.array(probas[i*test_size:(i+1)*test_size])[:group_size]
        #p = np.random.permutation(len(batch))
        #indices = p[np.argpartition(batch[p], -k)[-k:]]
        indices = np.argpartition(batch, -k)[-k:]
        max_idx = np.argmax(batch) + i*test_size
        html = '<table><tr>'
        html += '<td></td>'
        html += '<td>%s</td></tr>' % '<br/>'.join(to_words(test_data['c'][i*test_size]).split('__eot__'))
        for j in range(i*test_size, (i+1)*test_size):
            html += '<tr><td>%.2f</td>' % batch[j-i*test_size]
            html += '<td>'
            r = to_words(test_data['r'][j])
            if max_idx == j:
                html += '<b>' + r + '</b>'
            else:
                html += r
            html += '</td></tr>'
        html += '</table><br/><br/>'
        if 0 in indices:
            L_correct.append(i)
            html_correct += html
        else:
            L_incorrect.append(i)
            html_incorrect += html
    return L_correct, L_incorrect, html_correct, html_incorrect

_, _, html_correct, _ = generate_report(test_probas, k=1)
_, _, _, html_incorrect = generate_report(test_probas, k=5)
with open('ubuntu_correct.html', 'wb') as f:
    f.write('%s\n' % html_correct)
with open('ubuntu_errors.html', 'wb') as f:
    f.write('%s\n' % html_incorrect)


