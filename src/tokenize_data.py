import csv
import joblib
#from jnius import autoclass

TRAIN_FILE = 'data/trainset.csv'
VAL_FILE = 'data/valset.csv'
TEST_FILE = 'data/testset.csv'
#Twokenizer = autoclass('cmu.arktweetnlp.Twokenize')

vocab = set()

def tokenize(s):
#def tokenize(s, tokenizer=Twokenizer()):
    return s.split()
    s = ' '.join(s.split('</s>'))
    tokens = tokenizer.tokenizeRawTweetText(s.decode('utf-8'))
    return [tokens.get(i) for i in xrange(tokens.size())]

def tokenize_file(fname):
    print fname
    writer = csv.writer(open(fname + '.tok', 'wb'))
    with open(fname, 'rb') as f:
        reader = csv.reader(f)
        for i,row in enumerate(reader):
            if i == 0:
                writer.writerow(row)
                continue
            if i % 10000 == 0:
                print i
            context, response, label = row[:3]

            context = context.replace('</s>', '')
            response = response.replace('</s>', '')

            tok_context = tokenize(context)
            tok_response = tokenize(response)

            """
            context = ' '.join(tok_context)
            response = ' '.join(tok_response)
            """

            for w in tok_context + tok_response:
                vocab.add(w)

#            writer.writerow([context, response, label])

def main():
    tokenize_file(TRAIN_FILE)
    tokenize_file(VAL_FILE)
    tokenize_file(TEST_FILE)
    joblib.dump(vocab, 'vocab.pkl')

if __name__ == '__main__':
    main()
