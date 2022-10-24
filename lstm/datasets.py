"""
    Costumized datasets
"""

import torch
from torch.utils.data import Dataset

# costumized SubjectivityDataset object overwriting the Dataset Class
class SubjectivityDataset(Dataset):
    # w2id is a dictionary for mapping each word into an unique id, we create a numeric corpus
    def __init__(self, x, y, w2id):
        super().__init__()
        self.corpus = self.mapping(x, w2id)
        self.labels = y

    def __getitem__(self, idx):
        return torch.tensor(self.corpus[idx]), self.labels[idx] 
        
    def __len__(self):
        return len(self.labels)

    # map each word in corpus to its Id, in that way we can return a torch tensor type
    def mapping(self, x, w2id):
        corpus = []
        # iterate through sentences
        for sent in x:
            id_sent = []
            # iterate through words in sentences
            for w in sent:
                id = w2id[w] if w in w2id.keys() else w2id["unk"]
                id_sent.append(id)
            # append each mapped sentence to the new numeric corpus
            corpus.append(id_sent)
        return corpus

# costumized PolarityDataset object overwriting the Dataset Class
class PolarityDataset(Dataset):
    # w2id is a dictionary for mapping each word into an unique id, we create a numeric corpus
    def __init__(self, x, y, w2id):
        super().__init__()
        self.corpus = self.mapping(x, w2id)
        self.labels = y

    def __getitem__(self, idx):
        return torch.tensor(self.corpus[idx]), self.labels[idx] 
    
    def __len__(self):
        return len(self.labels)

    # map each word in corpus to its Id, in that way we can return a torch tensor type
    def mapping(self, x, w2id):
        corpus = []
        # iterate through docs
        for doc in x:
            id_doc = []
            # iterate through sentences in dic
            for sent in doc:
                # iterate through words in sentences
                for w in sent:
                    id = w2id[w] if w in w2id.keys() else w2id["unk"]
                    id_doc.append(id)
                # append each mapped sentence to the new numeric corpus
            corpus.append(id_doc)
        return corpus