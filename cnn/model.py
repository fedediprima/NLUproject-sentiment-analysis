"""
    Model for sentiment analysis with Convolutional Neural Network
"""

import sys
import os
import torch.nn as nn
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parameters import EMBEDDING_DIM_CNN, N_FILTERS, FILTER_SIZES

class SentimentCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=EMBEDDING_DIM_CNN, n_filters=N_FILTERS, filter_sizes=FILTER_SIZES):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # convolutions with n different types of filter (specified in FILTER_SIZES) 
        self.convs = nn.ModuleList([ nn.Conv2d(in_channels = 1, out_channels = n_filters, kernel_size = (fs, embedding_dim)) for fs in filter_sizes])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, out_features = 1)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, text):
        # embeddings
        embeds= self.embedding(text).unsqueeze(1)

        # convolutions to our embeddings with the filters (sizes and number specified on parameters file)
        convolved = [F.relu(conv(embeds)).squeeze(3) for conv in self.convs]
            
        # max pooling to reduce parameters and help overfitting
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in convolved]
        
        # concatenation of all the pooled outputs
        output = torch.cat(pooled, dim = 1) 
        
        # dropout layer
        dropout = self.dropout(output)

        # fully-connected layer
        y_pred = self.fc(dropout)

        return y_pred.squeeze(-1)