"""
    Parameters
"""

import os

# Paths
ROOT_DIR_PATH = os.path.dirname(__file__)
BASELINE_DIR_PATH = os.path.join(ROOT_DIR_PATH, 'baseline')
LSTM_DIR_PATH = os.path.join(ROOT_DIR_PATH, 'lstm')
CNN_DIR_PATH = os.path.join(ROOT_DIR_PATH, 'cnn')
TRANS_DIR_PATH = os.path.join(ROOT_DIR_PATH, 'transformers')

# General parameters
PAD_TOKEN=0
LR_POL = 0.005
LR_SUBJ = 0.001
EPOCHS = 30
TEST_PER = 0.25
PATIENCE = 5
BATCH_SIZE = 256

# Parameters for LSTM
EMBEDDING_DIM_LSTM = 300
HIDDEN_DIM = 100

# Parameters for CNN
EMBEDDING_DIM_CNN = 150
N_FILTERS = 50
FILTER_SIZES = [2,3,4,5]

# Parameters for transformer
BATCH_SIZE_TRANS_SUBJ = 64
BATCH_SIZE_TRANS_POL = 8
LR_TRANS = 5e-5
SUBJ_TRANS_MODEL = "bert-base-uncased"
POL_TRANS_MODEL = "distilbert-base-uncased"
WD = 0.01
