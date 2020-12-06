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
#from plot import dynamicplot

EPOCH = 5
BATCH_SIZE = 20
LR = 0.01
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
 
#NN 
#net=torch.nn.Sequential(
#    nn.Linear(28*28,30),
#    nn.ReLU()
#    nn.Linear(30,10)
#)

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
#optimizer=torch.optim.SGD(net.parameters(),lr=LR)
optimizer = ADAM(net.parameters())

#optimizer = GWDC(GWDCnet.parameters())
#optimizers = [optimADAM, optimGWDC]
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
