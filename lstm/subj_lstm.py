"""
    The purpose of this file is to train a subjectivity classifier using LSTM
"""

import time
import torch
import os
import sys
import copy
import pickle
from nltk.corpus import subjectivity
from torch.utils.data import DataLoader
from datasets import SubjectivityDataset
from torch import optim
import torch.nn as nn
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import SentimentLSTM
from functions import count_parameters, word2id, collate_fn, init_weights, predict_subjectivity_LSTM, plot_loss_accuracy
from parameters import LR_SUBJ, EPOCHS, TEST_PER, PATIENCE, ROOT_DIR_PATH, LSTM_DIR_PATH, BATCH_SIZE
from train_functions_lstm import train_loop, eval_loop

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    #load the dataset
    obj = subjectivity.sents(categories='obj')
    subj = subjectivity.sents(categories='subj')

    # restrict the dataset if cuda not available
    if device != "cuda":
        obj = obj[:30]
        subj = obj[:30]

    # Compute lebels and split in train/test set
    labels = [0] * len(obj) + [1] * len(subj)
    X_train, X_test, y_train, y_test = train_test_split(obj+subj, labels, test_size=TEST_PER, random_state=1)

    # create the dictionary of the training set that maps words to unique ids with also 'unk' and pad token, we want a numeric corpus 
    voc = set([w for d in X_train for w in d])
    w2id = word2id(voc)

    # create the SubjectivityDataset object for training and test set overwriting the Dataset Class
    train_set = SubjectivityDataset(X_train, y_train, w2id)
    test_set = SubjectivityDataset(X_test, y_test, w2id)

    # Initialize dataloaders (collate_fn to handle tensors of different size)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn = collate_fn, num_workers=2)  
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, collate_fn = collate_fn)

    # Instantiate the model with some hyperparam, other parameters are specified in the parameters file
    bidirectional = True
    num_embedding = len(voc) + 2 # voc + 2 for the 0 padding e unk
    net = SentimentLSTM(num_embedding, bidirectional)
    net.apply(init_weights)
    net = net.to(device)
    print(f"Number of parameters in the net: {count_parameters(net)}")

    # set the optimizer and loss function
    optimizer = optim.Adam(net.parameters(), lr=LR_SUBJ)
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
        print(f"Time elapsed: {round(time_el, 3)}")

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
    plot_loss_accuracy(losses_train, losses_test, accuracies_train, accuracies_test)

    # save the weights and w2id of the best model
    if not os.path.exists(os.path.join(ROOT_DIR_PATH, LSTM_DIR_PATH, "weights")):
        os.mkdir(os.path.join(ROOT_DIR_PATH, LSTM_DIR_PATH, "weights"))
    torch.save(best_model.state_dict(), os.path.join(ROOT_DIR_PATH, LSTM_DIR_PATH, "weights/lstm_subj.pt"))
    with open(os.path.join(ROOT_DIR_PATH, LSTM_DIR_PATH, "weights/w2id_subj_lstm.pkl"), "wb") as tf:
        pickle.dump(w2id,tf)
    print("Model saved...")

    # some examples of subjectivity/objectivity prediction using our best model
    print("\nInput: ", " ".join(X_test[100]))
    print("Correct Label: ", y_test[100])
    predict_subjectivity_LSTM(X_test[100], best_model, w2id, device)

    print("Input: ", " ".join(X_test[40]))
    print("Correct Label: ", y_test[40])
    predict_subjectivity_LSTM(X_test[40], best_model, w2id, device)

if __name__ == "__main__":
    main()