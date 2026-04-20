#--------------------------
# day_1_test.py - Test Run
# Target: < 5 min on CPU
#---------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import gzip
import random 
import time 
import matplotlib
matplotlib.use('Agg') # non-interactive backend — saves to file, no display needed
import matplotlib.pyplot as plt 
from collections import defaultdict 

#--------------------------------
# Configuration - Hyperparameters
# --------------------------------

# --- Model Architecture ---
D_MODEL  = 64   # embedding dimension - small but not trivial 
N_HEADS  = 4    # attention heads - D_MODEL must be divisible by N_HEADS 
D_FF     = 256  #feed-forward hidden dimensions 
N_LAYERS = 2    # transformer blocks - 2 layers 
SEQ_LEN  = 128  # context window - covers 2 NCA steps (64 toeksn each) 

# --- 2D NCA Parameters --- 
GRID_H   = 16   # grid height (paper uses 12)
GRID_W   = 16   # grid width 
N_STATES = 4    # alphabet size - 4 states per cell
PATCH    = 2    # patch size - 2X2 patches

#vocab size = N_STATES^(PATCH*PATCH)  = 4^(2*2)=256 
NCA_VOCAB = N_STATES ** (PATCH * PATCH) #256

# --- 1D NCA data generation --- 
TAPE_LEN = 64   # length of 1D tape 
N1D_WIN  = 2    # window size for tokenization ( 2 cells -> 1 token) 
#vocab size = N_STATES^(PATCH*PATCH)  = 4^(2)=16 
NCA_VOCAB = N_STATES ** N1D_WIN #16


#--- NCA data generation ---
N_RULES   = 500 # unique NCA rules (=different training sequences) 
N_TRAJ    = 20  # trajectories per rule (different strating states) 
N_STEPS   = 64  # timesteps per trajectory 
NCA_EPOCHS = 3  # epochs over NCA data during pre-pre-training 

# --- Transfer task (DYCK-1) --- 
# Vocabulary: 0='(', 1=')', 2=PAD, 3=EOS 
DYCK_VOCAB   = 4 
DYCK_DEPTH   = 8  # max nesting depth 
DYCK_SEQ_LEN = 64 # length before padding 
N_DYCK_TRAIN = 10_000 
N_DYCK_VAL   = 2_000 
DYCK_EPOCHS  = 20 

# --- Training --- 
BATCH_SIZE   = 32 
NCA_LR       = 3e-4
DYCK_LR      = 1e-4

# --- Experiments --- 
SEEDS = [42, 43, 44] # 3 seeds -> means (+/-) std 
THRESHOLD = 0.3 # val loss threshold for "steps to convergence" metric 

print(f"NCA 2D vocab size : {NCA_VOCAB}")
print(f"NCA 1D vocab size : {NCA1D_VOCAB}") 
print(f"Dyck vocab size : {DYCK_VOCAB}") 
print(f"Total sequences : {N_RULES * N_TRAJ:,}") 
