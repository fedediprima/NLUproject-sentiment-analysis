"""
    Functions for training and evaluate the model
"""

import torch
from sklearn.metrics import accuracy_score, f1_score

# function for training the model
def train_loop(net, train_loader, optimizer, loss_fn, device):   
    cum_loss = 0
    acc = []
    f1 = []
    # set the net into train mode
    net.train()
    # loop on all the dataset, batch after batch
    for sample in train_loader:
        # zeroing the gradient
        optimizer.zero_grad() 

        # sample is a single batch: tensor(data), tensor(labels), tensor(lenghts)
        x, y, l = sample 

        # load data into GPU if available
        x = x.to(device)
        y = y.to(device)
        l = l.to(device)

        # forward pass
        y_pred = net(x) 

        # apply the loss
        loss = loss_fn(y_pred, y)
        
        # backward pass
        loss.backward()

        # update parameters
        optimizer.step()

        # update cumulative loss
        cum_loss += loss.item()

        # preproces the final prediction for return the statistics
        final_pred = torch.round(torch.sigmoid(y_pred))
        acc.append(accuracy_score(y.cpu().numpy(), final_pred.cpu().detach().numpy()))
        f1.append(f1_score(y.cpu().numpy(), final_pred.cpu().detach().numpy()))
    
    # process and return statistics
    avg_loss = cum_loss / len(train_loader)
    avg_acc = sum(acc) / len(acc)
    avg_f1 = sum(f1) / len(f1)

    return avg_loss, avg_acc, avg_f1

# function for evaluating the model
@torch.no_grad()
def eval_loop(net, test_loader, loss_fn, device):
    cum_loss = 0
    acc = []
    f1 = []
    # set the net into evaluate mode
    net.eval()
    with torch.no_grad():
        # loop on all the dataset, batch after batch
        for sample in test_loader:

            # sample is a single batch: tensor(data), tensor(labels), tensor(lenghts)
            x, y, l = sample 

            # load data into GPU if available
            x = x.to(device)
            y = y.to(device)
            l = l.to(device)

            # forward pass
            y_pred = net(x)

            # apply the loss
            loss = loss_fn(y_pred, y)

            # update cumulative loss
            cum_loss += loss.item()

            # preproces the final prediction for return the statistics
            final_pred = torch.round(torch.sigmoid(y_pred))
            acc.append(accuracy_score(y.cpu().numpy(), final_pred.cpu().numpy()))
            f1.append(f1_score(y.cpu().numpy(), final_pred.cpu().numpy()))
    
    # process and return statistics
    avg_loss = cum_loss / len(test_loader)
    avg_acc = sum(acc) / len(acc)
    avg_f1 = sum(f1) / len(f1)

    return avg_loss, avg_acc, avg_f1





