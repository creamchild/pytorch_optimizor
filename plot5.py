import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

IO = './result/exp5/'

  
files = os.listdir(IO)
true_files = []
for file in files:
    if '.txt' in file:
        true_files.append(file)
true_files = sorted(true_files)

train_losses = []
for t in true_files:
    train_loss = []
    with open(IO+t, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'Loss' in line:
                train_loss.append(float(line[line.find('s:')+2:line.find(', P')-1]))
    train_losses.append(train_loss)
    
train_ppls = []
for t in true_files:
    train_ppl = []
    with open(IO+t, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'Perplexity' in line:
                train_ppl.append(float(line[line.find('y:')+3:]))
    train_ppls.append(train_ppl)
    
epoch = np.arange(1,len(train_loss)+1,1)

label = []
for i in range(len(train_losses)):
    plt.plot(epoch,train_losses[i])
    label.append(true_files[i][:true_files[i].find('.')])
plt.xlabel('epoch')
plt.ylabel('training loss')
plt.xticks(range(1,len(epoch)+1))
plt.legend(label, loc='upper right')
plt.savefig(IO+'plot_loss.jpg')
plt.cla()
for i in range(len(train_ppls)):
    plt.plot(epoch,train_ppls[i])
plt.xlabel('epoch')
plt.ylabel('training ppl')
plt.xticks(range(1,len(epoch)+1))
plt.legend(label, loc='upper right')
plt.savefig(IO+'plot_ppl.jpg')
print('ok')