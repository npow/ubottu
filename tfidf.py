import csv
import joblib
import numpy as np
import scipy
from sklearn.feature_extraction.text import *
from sklearn.metrics import *
from sklearn.svm import *

TRAIN_FILE = 'data/trainset.csv'
VAL_FILE = 'data/valset.csv'
#TEST_FILE = 'data/testset.csv'

vocab = joblib.load('vocab.pkl')

def load_data(fname):
    print fname
    C, R, Y = [], [], []
    with open(fname, 'rb') as f:
        reader = csv.reader(f)
        for i,row in enumerate(reader):
            if i == 0:
                continue
            context, response, label = row[:3]
            C.append(context)
            R.append(response)
            Y.append(label)
    return C, R, Y

def main():
    C_train, R_train, Y_train = load_data(TRAIN_FILE)
    print "done loading train"
    vectorizer = TfidfVectorizer(vocabulary=list(vocab))
    vectorizer.fit(C_train)
    print "done fit"
    C_train = vectorizer.transform(C_train)
    R_train = vectorizer.transform(R_train)
    Y_train = np.array(Y_train).astype(np.int32)

    C_val, R_val, Y_val = load_data(VAL_FILE)
    print "done loading test"
    C_val = vectorizer.transform(C_val)
    R_val = vectorizer.transform(R_val)
    Y_val = np.array(Y_val).astype(np.int32)

    print C_train.shape, R_train.shape, Y_train.shape

    X_train = scipy.sparse.hstack([C_train, R_train])
    X_val = scipy.sparse.hstack([C_val, R_val])
    print X_train.shape, X_val.shape

    clf = LinearSVC()
    clf.fit(X_train, Y_train)
    pred = clf.predict(X_val)

    print classification_report(Y_val, pred)

if __name__ == '__main__':
    main()
