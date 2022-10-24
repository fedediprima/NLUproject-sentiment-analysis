"""
    The purpose of this file is to train a subjectivity classifier using TRANSOFRMERS
"""

import time
import torch
import os
import sys
import copy
from nltk.corpus import subjectivity
from torch.utils.data import DataLoader
from datasets import SubjectivityDataset
from torch import optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from transformers import BertForSequenceClassification

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functions import count_parameters, predict_subjectivity_TRANS, plot_loss_accuracy
from parameters import BATCH_SIZE_TRANS_SUBJ, LR_TRANS, EPOCHS, TEST_PER, PATIENCE, ROOT_DIR_PATH, TRANS_DIR_PATH, SUBJ_TRANS_MODEL
from train_functions_trans import train_loop, eval_loop

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    # load the dataset
    obj = subjectivity.sents(categories='obj')
    subj = subjectivity.sents(categories='subj')

    # restrict the dataset if cuda not available
    if device != "cuda":
        obj = obj[:30]
        subj = obj[:30]

    # Compute lebels and split in train/test set
    labels = [0] * len(obj) + [1] * len(subj)
    X_train, X_test, y_train, y_test = train_test_split(obj+subj, labels, test_size=TEST_PER, random_state=1)

    # pretrained bert tokenizer 
    tokenizer = BertTokenizer.from_pretrained(SUBJ_TRANS_MODEL)
    
    # create the SubjectivityDataset object for training and test set overwriting the Dataset Class, the tokenizer is inside the Class definition
    train_set = SubjectivityDataset(X_train, y_train, tokenizer)
    test_set = SubjectivityDataset(X_test, y_test, tokenizer)
    
    # Initialize dataloaders (no collate_fn, tensors are already the same size)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE_TRANS_SUBJ, shuffle=True)  
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE_TRANS_SUBJ)

    # Instantiate the bert pretraned model
    # BertForSequenceClassification is simply a BERT model with a linear layer for sentence classification (2 labels)
    bert_model = BertForSequenceClassification.from_pretrained(SUBJ_TRANS_MODEL, num_labels = 2)
    net = bert_model.to(device)
    print(f"Number of parameters in the net: {count_parameters(net)}")

    # set the optimizer (loss function is already inside the model)
    optimizer = optim.Adam(net.parameters(), lr=LR_TRANS)

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
        loss_train, acc_train, f1_train = train_loop(net, train_loader, optimizer, device)
        print(f"Training Loss : {round(loss_train , 4)} Training Accuracy : {round(acc_train*100, 2)} Training F1 score : {round(f1_train , 4)}")
        
        # list of losses and accuracies for plotting the results
        losses_train.append(loss_train)
        accuracies_train.append(acc_train)

        # evaluate the network
        loss_test, acc_test, f1_test = eval_loop(net, test_loader, device)
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
    loss_test, acc_test, f1_test = eval_loop(best_model, test_loader, device)
    print(f"Best Model statistics: \nLoss : {round(loss_test, 4)} Accuracy : {round(acc_test*100, 2)} F1 score : {round(f1_test , 4)}")
    
    # plot train and test progress during training
    plot_loss_accuracy(losses_train, losses_test, accuracies_train, accuracies_test)

    # save the weights of the best model
    print("Saving weights..")
    if not os.path.exists(os.path.join(ROOT_DIR_PATH, TRANS_DIR_PATH, "weights")):
        os.mkdir(os.path.join(ROOT_DIR_PATH, TRANS_DIR_PATH, "weights"))
    torch.save(best_model.state_dict(), os.path.join(ROOT_DIR_PATH, TRANS_DIR_PATH, "weights/trans_subj.pt"))
    print("Model saved")

    # some examples of subjectivity/objectivity prediction using our best model
    print("\nInput: ", " ".join(X_test[5]))
    print("Correct Label: ", y_test[5])
    predict_subjectivity_TRANS(X_test[5], best_model, tokenizer, device)

if __name__ == "__main__":
    main()
