"""
    Utiliy functions
"""

import torch
import sys
import os
import torch.nn as nn
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parameters import PAD_TOKEN, FILTER_SIZES

# mappping from words to integers (in order to use torch tensors)
def word2id(voc):
    w2id = {"<pad>": 0}
    for w in voc:
        w2id[w] = len(w2id)
    w2id["unk"] = len(w2id)
    return w2id

# add zero padding at the end of the sequence if needed (from batch * sent_len to batch * max_len)
def collate_fn(batch):
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths) == 0 else max(lengths)
        # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape batch_size X maximum length of a sequence
        padded_seqs = torch.LongTensor(len(sequences), max_len).fill_(PAD_TOKEN)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq 
        
        padded_seqs = padded_seqs.detach() 
        return padded_seqs, torch.LongTensor(lengths)

    # Sort batch by seq length
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    x = [x for x, y in batch]
    y = [y for x, y in batch]
    
    x, lengths = merge(x)
    y = torch.FloatTensor(y)

    return x, y, lengths

# from list of list to string
def lol2str(lol):
    return " ".join([w for sent in lol for w in sent])

# from list to string
def list2str(l):
    return ' '.join(w for w in l)

# flatten a list of list into a single list
def flatten(lol):
    return [item for l in lol for item in l]

# remove objective sentences from a document with NB classifier
def remove_objective_sents_NB(classifier, vectorizer, doc): 
    filt_sent = []
    doc = [list2str(p) for p in doc]
    vectors = vectorizer.transform(doc)
    subj_prediction = classifier.predict(vectors)
    # apppend only subjective sentence (subj_prediction = 1)
    for d, est in zip(doc, subj_prediction):
        if est==1:
            filt_sent.append(d)
    filt_doc = list2str(filt_sent)
    return filt_doc

# remove objective sentences from a document with LSTM classifier
def remove_objective_sents_LSTM(doc, model, w2id, device):
    model.eval()
    filt_doc = []
    for sent in doc:
        ids = []
        for w in sent:
            id = w2id[w] if w in w2id.keys() else w2id["unk"]
            ids.append(id)
        x = torch.LongTensor(ids).unsqueeze(dim=0).to(device)
        l = torch.LongTensor([len(ids)])
        y_pred = model(x, l).squeeze(dim=0)
        final_prediction = torch.round(torch.sigmoid(y_pred))
        # add only subjective sentence to the filtred doc
        if final_prediction == 1:
            filt_doc.append(sent) 
    return filt_doc

# remove objective sentences from a document with CNN classifier
def remove_objective_sents_CNN(doc, model, w2id, device):
    model.eval()
    filt_doc = []
    for sent in doc:
        ids = []
        for w in sent:
            id = w2id[w] if w in w2id.keys() else w2id["unk"]
            ids.append(id)
        # control the size of the tensor (can't apply 5x1 filter on a sentences of length 4)
        if len(ids)>max(FILTER_SIZES):
            x = torch.LongTensor(ids).unsqueeze(dim=0).to(device)
            y_pred = model(x).squeeze(dim=0)
            final_prediction = torch.round(torch.sigmoid(y_pred))
            # add only subjective sentence to the filtred doc
            if final_prediction == 1:
                filt_doc.append(sent) 
    return filt_doc

# remove objective sentences from a document with TRANS classifier
def remove_objective_sents_TRANS(doc, model, tokenizer, device):
    model.eval()
    filt_doc=[]
    for sent in doc:
      tokenized = tokenizer(" ".join(sent), return_tensors="pt", return_attention_mask = True, padding = True, truncation = True, return_length = True).to(device)
      x = tokenized["input_ids"]
      mask = tokenized["attention_mask"]
      y_pred = model(x, mask).logits
      final_prediction = torch.argmax(y_pred).cpu().numpy()
      if final_prediction == 1:
        filt_doc.append(sent) 
    return filt_doc

# randomly initialize the weights (reused from labs)
def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)

# count the parameters of the model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# predict subjectivity/objectivity of a text with LSTM model
def predict_subjectivity_LSTM(text, model, w2id, device):
    model.eval()
    ids = []
    for w in text:
        id = w2id[w] if w in w2id.keys() else w2id["unk"]
        ids.append(id)
    x = torch.LongTensor(ids).unsqueeze(dim=0).to(device)
    l = torch.LongTensor([len(ids)])
    y_pred = model(x, l).squeeze(dim=0)
    print("Model output: " , round(torch.sigmoid(y_pred).item(), 8))
    final_prediction = torch.round(torch.sigmoid(y_pred))
    if final_prediction == 0:
        label = "0 -> objective"
    else:
        label = "1 -> subjective"
    print("Prediction: " , label)

# predict subjectivity/objectivity of a text with CNN model
def predict_subjectivity_CNN(text, model, w2id, device):
    model.eval()
    ids = []
    for w in text:
        id = w2id[w] if w in w2id.keys() else w2id["unk"]
        ids.append(id)
    x = torch.LongTensor(ids).unsqueeze(dim=0).to(device)
    y_pred = model(x).squeeze(dim=0)
    print("Model output: " , round(torch.sigmoid(y_pred).item(), 8))
    final_prediction = torch.round(torch.sigmoid(y_pred))
    if final_prediction == 0:
        label = "0 -> objective"
    else:
        label = "1 -> subjective"
    print("Prediction: " , label)

# predict subjectivity/objectivity of a text with TRANS pretrained model
def predict_subjectivity_TRANS(text, model, tokenizer, device):
    model.eval()
    tokenized = tokenizer(" ".join(text), return_tensors="pt", return_attention_mask = True, padding = True, truncation = True, return_length = True).to(device)
    x = tokenized["input_ids"]
    mask = tokenized["attention_mask"]
    y_pred = model(x, mask).logits
    final_prediction = torch.argmax(y_pred).cpu().numpy()
    if final_prediction == 0:
        label = "0 -> objective"
    else:
        label = "1 -> subjective"
    print("Prediction: " , label)

# predict polarity of a text with LSTM model
def predict_polarity_LSTM(doc, model, w2id, device):
    model.eval()
    ids = []
    for sent in doc:
        for w in sent:
            id = w2id[w] if w in w2id.keys() else w2id["unk"]
            ids.append(id)
    x = torch.LongTensor(ids).unsqueeze(dim=0).to(device)
    l = torch.LongTensor([len(ids)])
    y_pred = model(x, l).squeeze(dim=0)
    print("Model output: " , round(torch.sigmoid(y_pred).item(), 8))
    final_prediction = torch.round(torch.sigmoid(y_pred))
    if final_prediction == 0:
        label = "0 - > negative"
    else:
        label = "1 -> positive"
    print("Prediction: " , label)

# predict polarity of a text with CNN model
def predict_polarity_CNN(doc, model, w2id, device):
    model.eval()
    ids = []
    for sent in doc:
        for w in sent:
            id = w2id[w] if w in w2id.keys() else w2id["unk"]
            ids.append(id)
    x = torch.LongTensor(ids).unsqueeze(dim=0).to(device)
    y_pred = model(x).squeeze(dim=0)
    print("Model output: " , round(torch.sigmoid(y_pred).item(), 8))
    final_prediction = torch.round(torch.sigmoid(y_pred))
    if final_prediction == 0:
        label = "0 - > negative"
    else:
        label = "1 -> positive"
    print("Prediction: " , label)

# predict polarity of a text with TRANS pretrained model
def predict_polarity_TRANS(text, model, tokenizer, device):
    model.eval()
    tokenized = tokenizer(" ".join(flatten(text)), return_tensors="pt", return_attention_mask = True, padding = True, truncation = True, return_length = True).to(device)
    x = tokenized["input_ids"]
    mask = tokenized["attention_mask"]
    y_pred = model(x, mask).logits
    final_prediction = torch.argmax(y_pred).cpu().numpy()
    if final_prediction == 0:
        label = "0 -> negative"
    else:
        label = "1 -> positive"
    print("Prediction: " , label)

# function to plot train and test loss and accuracy
def plot_loss_accuracy(losses_train, losses_test, accuracies_train, accuracies_test):
    fig = plt.figure(figsize=(8,8))
    loss = fig.add_subplot(2,1,1)
    loss.plot(losses_train, label='train loss')
    loss.plot(losses_test, label='test loss')
    plt.legend()
    loss.set_xlabel('epochs')
    loss.set_ylabel('loss')
    acc = fig.add_subplot(2,1,2)
    acc.plot(accuracies_train, label='train accuracy')
    acc.plot(accuracies_test, label='test accuracy')
    plt.legend()
    acc.set_xlabel('epochs')
    acc.set_ylabel('accuracy')
    plt.show()

# get n max values from a dict
def nbest(d, n=1):
    return dict(sorted(d.items(), key=lambda item: item[1], reverse=True)[:n])    