"""
    Model for sentiment analysis with Long Short Term Memory 
"""

import torch.nn as nn
import sys
import os
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parameters import EMBEDDING_DIM_LSTM, HIDDEN_DIM

# create my own model overwriting pytorch Module class
class SentimentLSTM(nn.Module):
   def __init__(self, num_embeddings, bidirectional, embedding_dim = EMBEDDING_DIM_LSTM, hidden_dim = HIDDEN_DIM):
      super().__init__()
      self.hidden_dim = hidden_dim
      self.embedding = nn.Embedding(num_embeddings, embedding_dim)
      self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, bidirectional=bidirectional, dropout=0.5, batch_first=True)
      self.dropout = nn.Dropout(0.5)
      self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, out_features = 1)

   def forward(self, x, l):
      # embeddings and lstm_out
      embeds = self.embedding(x)
      
      # dropout layer
      dropout = self.dropout(embeds)
      
      # packs a Tensor containing padded sequences of variable length
      packed_input = pack_padded_sequence(dropout, l.cpu(), batch_first=True, enforce_sorted=False)
      packed_output, (hidden, cell) = self.lstm(packed_input)
      output, _ = pad_packed_sequence(packed_output, batch_first=True)

      # FIRST METHOD, USE THE HIDDEN REPRESENTATION FOR CLASSIFICATION

      # context vector which are the last hidden states of the LSTM, we want the final layer forward and backward hidden states
      # #hidden = [num layers * num directions, batch size, hid dim]
      #if self.lstm.bidirectional:
      #   context_vector = torch.cat([hidden[-1,:,:], hidden[-2,:,:]], dim=1)
      #else:
      #  context_vector = hidden[-1,:]


      # SECOND METHOD, USE THE SUMMED LSTM OUTPUT FOR CLASSIFICATION (better performances)

      # output = [batch size, seq len, hidden dim * n directions]
      # the output is summed up and sent to the classifier (fully connected)
      context_vector = output.sum(dim=1)
      
      # dropout layer
      output = self.dropout(context_vector)
      
      # fully-connected layer
      y_pred = self.fc(output)
      
      return y_pred.squeeze(-1)
