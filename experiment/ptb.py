import sys 
sys.path.append(r'../')
import torch
import torch.nn as nn
import os
import argparse
import numpy as np
from torch.nn.utils import clip_grad_norm_
from utils import Dictionary, Corpus
from torchnlp.datasets import penn_treebank_dataset
from optimizer import ADAM
from optimizer import GWDC


#param import

parser = argparse.ArgumentParser(description='PTB GRU with PyTorch')
parser.add_argument('--epoch', type=int, default=5,
                    help='number of epoch')
parser.add_argument('--optim', type=str, default='ADAM',
                    help='type of optimizer (ADAM, GWDC)')
parser.add_argument('--amsgrad', type=bool, default=False,
                    help='type of amsgrad (True, False)')
parser.add_argument('--optimswitch', type=str, default='C1',
                    help='type of optimizer param (C1,C2,C3,D1,D2,D3)')
args = parser.parse_args()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#dataset
ROOT = '../data/penn-treebank/'
if not os.path.exists(ROOT):
    penn_treebank_dataset(ROOT)
# Hyper-parameters
embed_size = 128
hidden_size = 1024
num_layers = 3
num_epochs = args.epoch
num_samples = 1000     # number of words to be sampled
batch_size = 20
seq_length = 30
lr = 0.001

# Load "Penn Treebank" dataset
corpus = Corpus()
ids = corpus.get_data(ROOT+'ptb.train.txt', batch_size)
test_ids = corpus.get_data(ROOT+'ptb.test.txt', batch_size)
vocab_size = len(corpus.dictionary)
num_batches = ids.size(1) // seq_length


# RNN based language model
class RNNLM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(RNNLM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x, h):
        # Embed word ids to vectors
        x = self.embed(x)
        
        # Forward propagate GRU
        out, h = self.gru(x)
        
        # Reshape output to (batch_size*sequence_length, hidden_size)
        out = out.reshape(out.size(0)*out.size(1), out.size(2))
        
        # Decode hidden states of all time steps
        out = self.linear(out)
        return out, h

model = RNNLM(vocab_size, embed_size, hidden_size, num_layers).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()

hook_lr = None
hook_beta = None
if args.optimswitch == 'C1':
    lr = 1e-3
if args.optimswitch == 'C2':
    lr = 1e-3
    hook_beta = lambda g, n: g['lr']
if args.optimswitch == 'C3':
    lr = 1e-2
    hook_beta = lambda g, n: g['lr']
if args.optimswitch == 'D1':
    lr = 1
    hook_lr = lambda g, n: g['lr'] * (n ** -0.5)
    hook_beta = lambda g, n: g['lr'] * (2 ** (-n))
if args.optimswitch == 'D2':
    lr = 1e-3
    hook_lr = lambda g, n: g['lr'] * (n ** -0.5)
    hook_beta = lambda g, n: g['lr'] * (2 ** (-n))
if args.optimswitch == 'D3':
    lr = 1e-2
    hook_lr = lambda g, n: g['lr'] * (n ** -0.5)
    hook_beta = lambda g, n: g['lr'] * (2 ** (-n))
if args.optim == 'ADAM':
    optimizer = ADAM(model.parameters(),lr=lr,amsgrad = args.amsgrad)
if args.optim == 'GWDC':
    optimizer = GWDC(model.parameters(),lr=lr,amsgrad = args.amsgrad)
if hook_lr:
    optimizer.c_hook_lr = hook_lr
if hook_beta:
    optimizer.c_hook_beta = hook_beta
# Truncated backpropagation
def detach(states):
    return [state.detach() for state in states] 

# Train the model
print('Start train!')
for epoch in range(num_epochs):
    # Set initial hidden and cell states
    states = (torch.zeros(num_layers, batch_size, hidden_size).to(device),
              torch.zeros(num_layers, batch_size, hidden_size).to(device))
    
    for i in range(0, ids.size(1) - seq_length, seq_length):
        # Get mini-batch inputs and targets
        inputs = ids[:, i:i+seq_length].to(device)
        targets = ids[:, (i+1):(i+1)+seq_length].to(device)
        
        # Forward pass
        states = detach(states)
        outputs, states = model(inputs, states)
        loss = criterion(outputs, targets.reshape(-1))
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        step = (i+1) // seq_length
        
        
    for i in range(0, test_ids.size(1) - seq_length, seq_length):
        # Get mini-batch inputs and targets
        inputs = test_ids[:, i:i+seq_length].to(device)
        targets = test_ids[:, (i+1):(i+1)+seq_length].to(device)
        
        # Forward pass
        states = detach(states)
        outputs, states = model(inputs, states)
        test_loss = criterion(outputs, targets.reshape(-1))
     
        #if step % 100 == 0:
    print ('Epoch [{}/{}], Train_Loss: {:.4f}, Perplexity: {:5.2f}, Test_Loss: {:.4f}, Perplexity: {:5.2f}'
                   .format(epoch+1, num_epochs, loss.item(), np.exp(loss.item()),test_loss.item(), np.exp(test_loss.item())))
