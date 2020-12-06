import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description='PLOT')
parser.add_argument('--exp', type=int, default=1,
                    help='name of exp(exp1,exp2,exp3,exp4)')
args = parser.parse_args()

IO = './result/'
if args.exp == 1:
    IO += 'exp1/'
if args.exp == 2:
    IO += 'exp2/'
if args.exp == 3:
    IO += 'exp3/'
if args.exp == 4:
    IO += 'exp4/'
  
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
            if 'TRAIN_Loss' in line:
                train_loss.append(float(line[line.find(':')+2:line.find('|')-1]))
    train_losses.append(train_loss)
    
train_accs = []
for t in true_files:
    train_acc = []
    with open(IO+t, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'TRAIN_Acc' in line:
                train_acc.append(float(line[line.find('c:')+3:]))
    train_accs.append(train_acc)
    
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
for i in range(len(train_accs)):
    plt.plot(epoch,train_accs[i])
plt.xlabel('epoch')
plt.ylabel('training acc')
plt.xticks(range(1,len(epoch)+1))
plt.legend(label, loc='upper right')
plt.savefig(IO+'plot_acc.jpg')
print('ok')