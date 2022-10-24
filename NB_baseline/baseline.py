"""
    Subjectivity and Polarity classifier using simple Naive Bayes Classification
"""

import sys
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functions import list2str, lol2str, remove_objective_sents_NB 
from nltk.corpus import movie_reviews, subjectivity


# Subjectivity/Objectivity Identification
def subjectivity_classifier():

    # initialize classifier and vectorizer for subjectivity classification
    vectorizer = CountVectorizer()
    classifier = MultinomialNB()

    # data is divided into objective and subjective sentences
    obj = subjectivity.sents(categories='obj')
    subj = subjectivity.sents(categories='subj')

    # preprocess the dataset
    corpus = [list2str(d) for d in obj] + [list2str(d) for d in subj]
    vectors = vectorizer.fit_transform(corpus)
    labels = [0] * len(obj) + [1] * len(subj)

    # evaluation
    accuracy = cross_validate(classifier, vectors, labels, cv=StratifiedKFold(n_splits=10) , scoring=['accuracy'])
    avg_accuracy = sum(accuracy['test_accuracy'])/len(accuracy['test_accuracy'])

    f1_score = cross_validate(classifier, vectors, labels, cv=StratifiedKFold(n_splits=10) , scoring=['f1'])
    avg_f1_score = sum(f1_score['test_f1'])/len(f1_score['test_f1'])

    # train again the classifier to return
    classifier.fit(vectors, labels)

    return avg_accuracy, avg_f1_score, classifier, vectorizer


# Polarity Classification: positive or negative.
def polarity_classifier(subj_classifier, subj_vectorizer, filter):
   
    # initialize classifier and vectorizer for Polairty classification
    vectorizer = CountVectorizer()
    classifier = MultinomialNB()

    # data is divided into negative and positive sentences
    neg = movie_reviews.paras(categories='neg')
    pos = movie_reviews.paras(categories='pos')

    # remove objective sentences, better performances
    if filter:
        corpus = []
        doc = neg + pos
        for d in doc:
            corpus.append(remove_objective_sents_NB(subj_classifier, subj_vectorizer, d))
    else:
        corpus = [lol2str(d) for d in neg] + [lol2str(d) for d in pos]

    # preprocess the dataset
    vectors = vectorizer.fit_transform(corpus)
    labels = [0] * len(neg) + [1] * len(pos)

    # evaluation 
    accuracy = cross_validate(classifier, vectors, labels, cv=StratifiedKFold(n_splits=10) , scoring=['accuracy'])
    avg_accuracy = sum(accuracy['test_accuracy'])/len(accuracy['test_accuracy'])

    f1_score = cross_validate(classifier, vectors, labels, cv=StratifiedKFold(n_splits=10) , scoring=['f1'])
    avg_f1_score = sum(f1_score['test_f1'])/len(f1_score['test_f1'])
    
    return avg_accuracy, avg_f1_score


def main(subj_filter):
    
    print("Baseline:")

    # subjectivity classifier using simple Naive Bayes Classification
    subj_accuracy, subj_f1, subj_classifier, subj_vectorizer = subjectivity_classifier()
    print(f"Accuracy on Naive Bayes subjectivity classification: {round(subj_accuracy*100, 3)}")
    print(f"F1 score on Naive Bayes subjectivity classification: {round(subj_f1, 3)}")

    # polarity classifier using simple Naive Bayes Classification and removing objective sentences (filter set to True by default)
    polarity_accuracy, polarity_f1 = polarity_classifier(subj_classifier, subj_vectorizer, subj_filter)
    print(f"Accuracy on Naive Bayes polarity classification: {round(polarity_accuracy*100, 3)}")
    print(f"F1 score on Naive Bayes polarity classification: {round(polarity_f1, 3)}")

if __name__ == "__main__":
    main(subj_filter = True)