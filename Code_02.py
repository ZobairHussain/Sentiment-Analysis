import nltk
import pickle
import time
import numpy as np
from sklearn.model_selection import KFold

start_time = time.time()
documents_f = open("pickled_algos/documents.pickle", "rb")
documents = pickle.load(documents_f)
documents_f.close()

word_features5k_f = open("pickled_algos/word_features5k.pickle", "rb")
word_features = pickle.load(word_features5k_f)
word_features5k_f.close()

featuresets_f = open("pickled_algos/featuresets5k.pickle", "rb")
featuresets = pickle.load(featuresets_f)
featuresets_f.close()
print("--- %s seconds ---" % (time.time() - start_time))
start_time = time.time()
print(len(featuresets))

kf = KFold(n_splits=10)
sum = 0
n = 0
for train, test in kf.split(featuresets):
    train_data = np.array(featuresets)[train]
    test_data = np.array(featuresets)[test]
    classifier = nltk.NaiveBayesClassifier.train(train_data)
    n = nltk.classify.accuracy(classifier, test_data)
    print("Accuracy :",n)
    sum +=n
    print("--- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
average = sum/10
print("Average: ",average*100)

