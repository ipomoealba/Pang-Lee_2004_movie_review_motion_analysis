import nltk
import re
import nltk.corpus
import random
import os
import codecs
import math
import numpy as np

from nltk.corpus import RegexpTokenizer


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


def unusual_words(text):
    """
    Detect typo
    """
    text_vocab = set(w.lower() for w in text if w.isalpha())
    english_vocab = set(w.lower() for w in nltk.corpus.words.words())
    unusual = text_vocab - english_vocab
    return sorted(unusual)


def buckets(filename, bucketName, separator, classColumn):

    numberOfBuckets = 10
    data = {}
    with open(filename) as f:
        lines = f.readlines()
    for line in lines:
        if separator != '\t':
            line = line.replace(separator, '\t')
        category = line.split()[classColumn]
        data.setdefault(category, [])
        data[category].append(line)
    buckets = []
    for i in range(numberOfBuckets):
        buckets.append([])
    for k in data.keys():
        random.shuffle(np.array(data[k]))
        bNum = 0
        for item in data[k]:
            buckets[bNum].append(item)
            bNum = (bNum + 1) % numberOfBuckets

    for bNum in range(numberOfBuckets):
        f = open("%s-%02i" % (bucketName, bNum + 1), 'w')
        for item in buckets[bNum]:
            f.write(item)
        f.close()


class BayesText:

    def __init__(self, trainingdir="training_data", stopwordlist=nltk.corpus.stopwords.words()):
        """
        Bayers Cluster
        """
        self.vocabulary = {}
        self.prob = {}
        self.totals = {}
        self.stopwords = stopwordlist
        self.categories = [filename for filename in os.listdir(trainingdir)
                           if os.path.isdir(trainingdir + filename)]
        print("Counting ...")
        for category in self.categories:
            print('    ' + category)
            (self.prob[category],
             self.totals[category]) = self.train(trainingdir, category)

        toDelete = []
        for word in self.vocabulary:
            if self.vocabulary[word] < 3:
                toDelete.append(word)

        for word in toDelete:
            del self.vocabulary[word]

        vocabLength = len(self.vocabulary)
        print("Computing probabilities:")
        for category in self.categories:
            #             print('    ' + category)
            denominator = self.totals[category] + vocabLength
            for word in self.vocabulary:
                if word in self.prob[category]:
                    count = self.prob[category][word]
                else:
                    count = 1
                self.prob[category][word] = (float(count + 1)
                                             / denominator)
        print ("DONE TRAINING\n\n")

    def train(self, trainingdir, category):
        currentdir = trainingdir + category
        files = os.listdir(currentdir)
        counts = {}
        total = 0
        for file in files:
            print(currentdir + '/' + file)
            f = codecs.open(currentdir + '/' + file, 'r', 'iso8859-1')
            for line in f:
                tokens = split_text(line)
                for token in tokens:
                    if token != '' and not token in self.stopwords:
                        self.vocabulary.setdefault(token, 0)
                        self.vocabulary[token] += 1
                        counts.setdefault(token, 0)
                        counts[token] += 1
                        total += 1
            f.close()
        return(counts, total)

    def classify(self, filename):
        results = {}
        for category in self.categories:
            results[category] = 0
        f = codecs.open(filename, 'r', 'iso8859-1')
        for line in f:
            tokens = line.split()
            for token in tokens:
                if token in self.vocabulary:
                    for category in self.categories:
                        if self.prob[category][token] == 0:
                            print("%s %s" % (category, token))
                        results[category] += math.log(
                            self.prob[category][token])
        f.close()
        results = list(results.items())
        results.sort(key=lambda tuple: tuple[1], reverse=True)
        return results[0][0]

    def testCategory(self, directory, category):
        files = os.listdir(directory)
        total = 0
        correct = 0
        for file in files:
            total += 1
            result = self.classify(directory + file)
            if result == category:
                correct += 1
        return (correct, total)

    def test(self, testdir):
        categories = os.listdir(testdir)
        categories = [filename for filename in categories if
                      os.path.isdir(testdir + filename)]
        correct = 0
        total = 0
        for category in categories:
            (catCorrect, catTotal) = self.testCategory(
                testdir + category + '/', category)
            correct += catCorrect
            total += catTotal
        print("\n\nAccuracy is  %f%%  (%i test instances)" %
              ((float(correct) / total) * 100, total))

if __name__ == "__main__":
    bt = BayesText("./txt_sentoken/")
    print(bt.prob)