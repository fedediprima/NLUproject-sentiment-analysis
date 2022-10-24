"""
    Costumized datasets
"""

from torch.utils.data import Dataset

# costumized SubjectivityDataset object overwriting the Dataset Class
class SubjectivityDataset(Dataset):
    def __init__(self, x, y, tokenizer):
        super().__init__()
        tokenized = tokenizer([" ".join(d) for d in x], return_tensors="pt", return_attention_mask = True, padding=True, truncation = True, return_length= True)
        self.corpus = tokenized["input_ids"]
        self.att_mask = tokenized["attention_mask"]
        self.length = tokenized["length"].numpy()
        self.labels = y

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.corpus[index], self.att_mask[index], self.labels[index], self.length[index]

# costumized PolarityDataset object overwriting the Dataset Class
class PolarityDataset(Dataset):
    def __init__(self, x, y, tokenizer):
        super().__init__()
        # tokenize with max length 512 words
        tokenized = tokenizer([self.lol2str(d) for d in x], return_tensors="pt", return_attention_mask = True, padding=True, truncation=True, max_length=512, return_length= True)
        self.corpus = tokenized["input_ids"]
        self.att_mask = tokenized["attention_mask"]
        self.length = tokenized["length"].numpy()
        self.labels = y

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.corpus[index], self.att_mask[index], self.labels[index], self.length[index]
        
    def lol2str(self, lol):
        return " ".join([w for sent in lol for w in sent])
