#%matplotlib notebook
import sys 
sys.path.append(r'../')
import torch.nn as nn
import torch
import torchvision
from torch.utils.data import DataLoader
from optimizer import GWDC
from optimizer import ADAM
import os
import math
import numpy as np
import argparse
#from plot import dynamicplot

#param import

parser = argparse.ArgumentParser(description='MNIST NN with PyTorch')
parser.add_argument('--epoch', type=int, default=5,
                    help='number of epoch')
parser.add_argument('--optim', type=str, default='ADAM',
                    help='type of optimizer (ADAM, GWDC)')
parser.add_argument('--amsgrad', type=bool, default=False,
                    help='type of amsgrad (True, False)')
parser.add_argument('--nntype', type=int, default=1,
                    help='type of NN (1, 2)')
parser.add_argument('--optimswitch', type=str, default='C1',
                    help='type of optimizer param (C1,C2,C3,D1,D2,D3)')
args = parser.parse_args()

EPOCH = args.epoch
BATCH_SIZE = 20
ROOT = "../data/mnist"
DOWNLOAD_MNIST = False
if not os.path.exists(ROOT):
    DOWNLOAD_MNIST = True
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print('Device is '+torch.cuda.get_device_name(0))
else:
    DEVICE = torch.device('cpu')
    print('Device is cpu')

trans=torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5],[0.5])
    ]
)


#dataloader
train_data=torchvision.datasets.MNIST(
    root=ROOT,
    train=True,
    transform=trans,
    download=DOWNLOAD_MNIST
)

train_loader=DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True)
 
test_data=torchvision.datasets.MNIST(
    root=ROOT,
    train=False,
    transform=trans,
    download=DOWNLOAD_MNIST
)
test_loader=DataLoader(test_data,batch_size=len(test_data),shuffle=False)

#choose NN type
if args.nntype == 1:
    #NN 1
    net=torch.nn.Sequential(
        nn.Linear(28*28,30),
        nn.ReLU(),
        nn.Linear(30,10)
    )
if args.nntype == 2:
    #NN 2
    net=torch.nn.Sequential(
        nn.Linear(28*28,30),
        nn.ReLU(),
        nn.Linear(30,30),
        nn.ReLU(),
        nn.Linear(30,10)
    )
net = net.to(DEVICE)
#net.to(DEVICE)

loss_function=nn.CrossEntropyLoss()

#set optimswitch
lr = 1e-3
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
    optimizer = ADAM(net.parameters(),lr=lr,amsgrad=args.amsgrad)
if args.optim == 'GWDC':
    optimizer = GWDC(net.parameters(),lr=lr,amsgrad=args.amsgrad)
if hook_lr:
    optimizer.c_hook_lr = hook_lr
if hook_beta:
    optimizer.c_hook_beta = hook_beta
print("Start training")
#DP = dynamicplot()
#DP.plotdefine()
for ep in range(EPOCH):
    batch_num = 0
    train_loss = 0
    test_batch_num = 0
    test_loss = 0
    train_num_correct=0
    test_num_correct=0
    for data in train_loader:#for every batch
        
        img,label=data
        img = img.view(img.size(0), -1)
         
        
        img=img.to(DEVICE)
        label=label.to(DEVICE)
 
        out=net(img)
        loss=loss_function(out,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        batch_num += 1
        _,prediction=torch.max(out,1)
        train_num_correct+=(prediction==label).sum()
    print('EPOCH: %d/%d'%(ep+1,EPOCH))
    train_accuracy=train_num_correct.cpu().numpy()/len(train_data)
    print('TRAIN_Loss: %.4f |  TRAIN_Acc: %.4f'%(train_loss/batch_num,train_accuracy))
    
    for data in test_loader:
        img,label=data
        img = img.view(img.size(0), -1)
 
        img=img.to(DEVICE)
        label=label.to(DEVICE)
 
        out=net(img)
        loss=loss_function(out,label)
        test_loss += loss.item()
        test_batch_num += 1
        _,prediction=torch.max(out,1)
        test_num_correct+=(prediction==label).sum()
    accuracy=test_num_correct.cpu().numpy()/len(test_data)
    print("TEST_Loss: %.4f |  TEST_Acc: %.4f"%(test_loss/test_batch_num,accuracy))
    #DP.showplot(train_loss1/batch_num,train1_accuracy,train_loss2/batch_num,train2_accuracy)
