import nltk
import re
import nltk.corpus
import random
import os
import codecs
import math
import numpy as np
import numpy

from nltk.corpus import RegexpTokenizer
from sklearn import metrics
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import KFold


def split_text(data, stopword_definder=True):
    """
    Spilt sentence to tokens and remove stopwords and punctuations
    And lemmeatize them
    """
    lemma = nltk.wordnet.WordNetLemmatizer()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(str(data))
    punctuation = re.compile(r'[-.?!,":;()|0-9]')
    stopwords = nltk.corpus.stopwords
    if stopwords:
        filtered_words = np.array([lemma.lemmatize(punctuation.sub("", token.lower())) for token in tokens
                                   if token.lower() not in stopwords.words('english') and len(token) > 2])
    else:
        filtered_words = np.array([lemma.lemmatize(
            punctuation.sub("", token.lower())) for token in tokens])
    return filtered_words


def input_dataset(traing_data, testing_data):
    train_words = []
    train_tags = []
    test_words = []
    test_tags = []
    stopwords = nltk.corpus.stopwords.words()
    categories = lambda fpath: [filename for filename in os.listdir(fpath)]

    def run_data_set(dataset, words, tags):
        # print(dataset)
        # print(words)
        for c in categories(dataset):
            current_dir = dataset + c
            try:
                files = os.listdir(current_dir)
                for file in files:
                    print(current_dir + file)
                    f = codecs.open(current_dir + '/' + file, 'r', 'iso8859-1')
                    for line in f:
                        words.append(line)
                        tags.append(c)
            except:
                print("Not Directory")
    run_data_set(traing_data, train_words, train_tags)
    run_data_set(testing_data, test_words, test_tags)
    return train_words, train_tags, test_words, test_tags


def vectorize(train_words, test_words):
    v = HashingVectorizer(tokenizer=split_text,
                          n_features=30000, non_negative=True)
    train_data = v.fit_transform(train_words)
    test_data = v.fit_transform(test_words)
    return train_data, test_data


def evaluate(actual, pred):
    m_precision = metrics.precision_score(actual, pred, pos_label="yes")
    m_recall = metrics.recall_score(actual, pred, pos_label="yes")
    print('precision:{0:.3f}'.format(m_precision))
    print('recall:{0:0.3f}'.format(m_recall))


def train_clf(train_data, train_tags):
    clf = MultinomialNB(alpha=0.01)
    clf.fit(train_data, numpy.asarray(train_tags))
    return clf

if __name__ == "__main__":
    train_file = "./txt_sentoken/testing_data/"
    test_file = "./txt_sentoken/testing_data/"
    train_words, train_tags, test_words, test_tags = input_dataset(
        train_file, test_file)
    train_data, test_data = vectorize(train_words, test_words)
    clf = train_clf(train_data, train_tags)
    pred = clf.predict(test_data)
    # evaluate(numpy.asarray(test_tags), pred)
