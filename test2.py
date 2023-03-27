import nltk
import random
from nltk.corpus import movie_reviews
import pickle
from nltk.classify.scikitlearn import SklearnClassifier

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB

documents = [(list(movie_reviews.words(fileid)), category)
            for category in movie_reviews.categories()
            for fileid in movie_reviews.fileids(category)
        ]

l_words = [word.lower() for word in movie_reviews.words()]

freq_dist = nltk.FreqDist(l_words)
#print(freq_dist.get('stupid'))


word_features = list(freq_dist.keys())[:3000]

def find_features(document):
    words = set(document)
    return {
            word:(word in words) for word in word_features
        }


#print(find_features(movie_reviews.words('neg/cv000_29416.txt')))


feature_sets = [(find_features(rev), category) for rev, category in documents]

training_set = feature_sets[:1900]
testing_set = feature_sets[1900:]


import os.path as path

classifier_file = 'naive_bayes_classifier.pickle'
if path.isfile(classifier_file):
    with open(classifier_file, 'rb') as file:
        classifier = pickle.load(file)
else:
    classifier = nltk.NaiveBayesClassifier.train(training_set)

accuracy = nltk.classify.accuracy(classifier, testing_set)
print(f"accuracy {accuracy}")
classifier.show_most_informative_features(15)


with open(classifier_file, 'wb') as file:
    pickle.dump(classifier, file)

