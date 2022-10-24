"""
    The purpose of this file is to train a polarity classifier using CNN
"""

import time
import torch
import os
import sys
import copy
import pickle
from nltk.corpus import movie_reviews
from torch.utils.data import DataLoader
from datasets import PolarityDataset
from torch import optim
import torch.nn as nn
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import SentimentCNN
from functions import count_parameters, word2id, collate_fn, init_weights, predict_polarity_CNN, plot_loss_accuracy, remove_objective_sents_CNN
from parameters import LR_POL, EPOCHS, TEST_PER, PATIENCE, ROOT_DIR_PATH, CNN_DIR_PATH, BATCH_SIZE
from train_functions_cnn import train_loop, eval_loop

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(subj_filter):
    # load the dataset
    neg = movie_reviews.paras(categories='neg')
    pos = movie_reviews.paras(categories='pos')

    # restrict the dataset if cuda not available
    if device != "cuda":
        pos = pos[:30]
        neg = neg[:30]
    
    # filter flag to decide whether or not remove objective sentence
    if subj_filter:
        # Load w2id subjectivity dictionary
        with open(os.path.join(ROOT_DIR_PATH, CNN_DIR_PATH, "weights/w2id_subj_cnn.pkl"), 'rb') as f:
            w2id_subj = pickle.load(f)
        
        # initialize and load trained subjectivity detector on cuda device
        subj_classifier = SentimentCNN(len(w2id_subj))
        subj_classifier.load_state_dict(torch.load(os.path.join(ROOT_DIR_PATH, CNN_DIR_PATH, "weights/cnn_subj.pt")))
        subj_classifier.to(device)

        # filter the negative and positive reviews from objective sentences
        subj_pos = []
        for review in pos:
            filt_review = remove_objective_sents_CNN(review, subj_classifier, w2id_subj, device)
            if filt_review:
                subj_pos.append(filt_review)

        subj_neg = []
        for review in neg:
            filt_review = remove_objective_sents_CNN(review, subj_classifier, w2id_subj, device)
            if filt_review:
                subj_neg.append(filt_review)
        
        # update pos and neg list with the filtred ones
        pos = subj_pos
        neg = subj_neg
    
    # Compute lebels and split in train/test set
    labels = [0] * len(neg) + [1] * len(pos)
    X_train, X_test, y_train, y_test = train_test_split(pos+neg, labels, test_size=TEST_PER, random_state=1)

    # create the dictionary of the training set that maps words to unique ids with also 'unk' and pad token, we want a numeric corpus
    voc = set([w for doc in X_train for sent in doc for w in sent])
    w2id = word2id(voc)
        
    # create the PolarityDataset object overwriting the Dataset Class
    train_set = PolarityDataset(X_train, y_train, w2id)
    test_set = PolarityDataset(X_test, y_test, w2id)

    # Initialize dataloaders (collate_fn to handle tensors of different size)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn = collate_fn, num_workers=2)  
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, collate_fn = collate_fn)

    # Instantiate the model with some hyperparam, other parameters are specified in the parameters file
    num_embedding = len(voc) + 2 # voc + 2 for the 0 padding e unk
    net = SentimentCNN(num_embedding)
    net.apply(init_weights)
    net = net.to(device)
    print(f"Number of parameters in the net: {count_parameters(net)}")

    # set the optimizer and loss function
    optimizer = optim.Adam(net.parameters(), lr=LR_POL)
    loss_fn = nn.BCEWithLogitsLoss()
    loss_fn = loss_fn.to(device)

    # initialize lists for plotting results
    losses_train = []
    losses_test = []
    accuracies_test = []
    accuracies_train = []

    # set variables for early stopping and for saving the model
    best_acc = 0
    patience = PATIENCE
    
    for i in range(EPOCHS):
        start = time.time()
        print(f"EPOCH {i+1}:")
        
        # train the network
        loss_train, acc_train, f1_train = train_loop(net, train_loader, optimizer, loss_fn, device)
        print(f"Training Loss : {round(loss_train , 4)} Training Accuracy : {round(acc_train*100, 2)} Training F1 score : {round(f1_train , 4)}")
        
        # list of losses and accuracies for plotting the results
        losses_train.append(loss_train)
        accuracies_train.append(acc_train)

        # evaluate the network
        loss_test, acc_test, f1_test = eval_loop(net, test_loader, loss_fn, device)
        print(f"Testing Loss : {round(loss_test , 4)} Testing Accuracy : {round(acc_test*100, 2)} Testing F1 score : {round(f1_test , 4)}")
        
        # list of losses and accuracies for plotting the results
        losses_test.append(loss_test)
        accuracies_test.append(acc_test)

        # print the time needed for each epoch
        time_el = time.time() - start 
        print(f"Time elapsed: {round(time_el, 3)} seconds")

        # save the model if test accuracy is the highest so far
        if acc_test > best_acc:
            best_acc = acc_test
            print("New best Model...")
            best_model = copy.deepcopy(net)
            patience = PATIENCE
        else:
            patience -= 1
        
        # early stopping if no improvements for 5 steps
        if patience <= 0:
            break

        print("\n")

    # print statistics of the best achieved model 
    loss_test, acc_test, f1_test = eval_loop(best_model, test_loader, loss_fn, device)
    print(f"Best Model statistics: \nLoss : {round(loss_test, 4)} Accuracy : {round(acc_test*100, 2)} F1 score : {round(f1_test , 4)}")
    
    # plot train and test progress during training
    plot_loss_accuracy(losses_train, losses_test, accuracies_test, accuracies_train)
    
    # save the weights and w2id of the best model
    if not os.path.exists(os.path.join(ROOT_DIR_PATH, CNN_DIR_PATH, "weights")):
        os.mkdir(os.path.join(ROOT_DIR_PATH, CNN_DIR_PATH, "weights"))
    torch.save(best_model.state_dict(), os.path.join(ROOT_DIR_PATH, CNN_DIR_PATH, "weights/cnn_pol.pt"))
    with open(os.path.join(ROOT_DIR_PATH, CNN_DIR_PATH, "weights/w2id_pol_cnn.pkl"), "wb") as tf:
        pickle.dump(w2id,tf)
    print("Model saved")

    # some examples of polarity prediction using our best model
    print("Input: ", X_test[100])
    print("Correct Label: ", y_test[100])
    predict_polarity_CNN(X_train[100], net, w2id, device)

if __name__ == "__main__":
    main(subj_filter = True)