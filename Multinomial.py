import pickle
import time
import nltk
import numpy as np
from sklearn.model_selection import KFold
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

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
print("--- %s seconds" % (time.time() - start_time))
start_time = time.time()

kf = KFold(n_splits=10)
sum = 0
n = 0
for train, test in kf.split(featuresets):
    train_data = np.array(featuresets)[train]
    test_data = np.array(featuresets)[test]
    LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
    LogisticRegression_classifier.train(train_data)
    n=(nltk.classify.accuracy(LogisticRegression_classifier, test_data))
    print("Accuracy :",n)
    sum +=n
    print("--- %s seconds" % (time.time() - start_time))
    start_time = time.time()
average = sum/10
print("Average of LogisticRegretion: ",average*100)
print()

kf = KFold(n_splits=10)
sum = 0
n = 0
for train, test in kf.split(featuresets):
    train_data = np.array(featuresets)[train]
    test_data = np.array(featuresets)[test]
    MNB_classifier = SklearnClassifier(MultinomialNB())
    MNB_classifier.train(train_data)
    n=(nltk.classify.accuracy(MNB_classifier, test_data))
    print("Accuracy :",n)
    sum +=n
    print("--- %s seconds" % (time.time() - start_time))
    start_time = time.time()
average = sum/10
print("Average of    MNB_classifier: ",average*100)
print()
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
    print("--- %s seconds" % (time.time() - start_time))
    start_time = time.time()
average = sum/10
print("Average    of   NaiveBayes: ",average*100)
print()

