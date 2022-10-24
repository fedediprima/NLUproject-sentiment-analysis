"""
    Dataset analysis
"""

import os
import sys
from nltk.corpus import subjectivity, movie_reviews
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functions import nbest, flatten

def main():
    # SUBJECTIVITY DATASET ANALYSIS
    # this dataset is a list of sentences, every sentence is classified as subjective or objective
    # list of lists of strings

    obj = subjectivity.sents(categories='obj')
    subj = subjectivity.sents(categories='subj')

    subj_dataset = obj + subj

    # length statistics
    print("Subjectivity dataset analysis: ")
    print("Dataset Length: ", len(subj_dataset))
    print("Objective sentences: ", len(obj))
    print("Subjective sentences: ", len(subj))

    # number of total words
    subj_dataset_words = [w.lower() for sent in subj_dataset for w in sent]
    print("Total number of words: ", len(subj_dataset_words))

    # lexicon of the dataset 
    subj_dataset_lex = set(subj_dataset_words)
    print("Vocabulary length: ", len(subj_dataset_lex))

    # frequency dictionary of the 20 most common words
    subj_dataset_freq = Counter(subj_dataset_words)
    print("Dataset frequency dictionary (20 best): \n", nbest(subj_dataset_freq, n=20))

    # words per sentence statistics
    l = [len(sent) for sent in subj_dataset]
    print("Maximum sentence's length: ", max(l))
    print("Minimum sentence's length: ", min(l))
    print("Average words per sentences: ", round(sum(l)/len(subj_dataset)))


    # MOVIE REVIEWS DATASET ANALYSIS
    # this dataset is a list of reviews, every reviews is a list of sentences and it is classified as positive or negative
    # list of lists of lists of strings

    neg = movie_reviews.paras(categories='neg')
    pos = movie_reviews.paras(categories='pos')

    reviews_dataset = neg + pos

    # length statistics
    print("\nMovie reviews dataset analysis: ")
    print("Dataset Length: ", len(reviews_dataset))
    print("Positive reviews: ", len(pos))
    print("Negative reviews: ", len(neg))

    # number of total words
    reviews_dataset_words = [w.lower() for review in reviews_dataset for sent in review for w in sent]
    print("Total number of words: ", len(reviews_dataset_words))

    # lexicon of the dataset 
    reviews_dataset_lex = set(reviews_dataset_words)
    print("Vocabulary length: ", len(reviews_dataset_lex))
    
    # frequency dictionary of the 20 most common words
    reviews_dataset_freq = Counter(reviews_dataset_words)
    print("Dataset frequency dictionary (20 best): \n", nbest(reviews_dataset_freq, n=20))

    # sentences per review statistics
    n_sentences = [len(review) for review in reviews_dataset]
    print("Maximum number of sentences per review: ", max(n_sentences))
    print("Minimum number of sentences per review: ", min(n_sentences))
    print("Average number of sentences per review: ", round(sum(n_sentences)/len(reviews_dataset)))

    # words per sentence statistics
    l = [len(sent) for review in reviews_dataset for sent in review]
    print("Maximum sentence's length: ", max(l))
    print("Minimum sentence's length: ", min(l))
    print("Average words per sentences: ", round(sum(l)/sum(n_sentences)))

    # words per review statistics
    w = [len(flatten(review)) for review in reviews_dataset]
    print("Maximum number of words per review: ", max(w))
    print("Minimum number of words per review: ", min(w))
    print("Average number of words per review: ", round(sum(w)/len(reviews_dataset)))
    
if __name__ == "__main__":
    main()