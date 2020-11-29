import matplotlib
import matplotlib.pyplot as plt
import time

class dynamicplot:
    
    def __init__(self):
        self.fig = plt.figure(figsize=(10, 4.8))
        self.ax_loss = self.fig.add_subplot(121)
        self.ax_acc = self.fig.add_subplot(122)
        self.Loss_list = []
        self.acc_list = []
    
    def plotdefine(self):
        plt.ion()
        self.fig.show()
        self.fig.canvas.draw()
        
    def showplot(self,loss,acc):
        self.loss = loss
        self.acc = acc
        #plot展示
        
        
        self.Loss_list.append(self.loss)
        self.acc_list.append(self.acc)
    
        self.ax_loss.clear()
        self.ax_loss.plot(self.Loss_list)
        # ax_loss.set_title()
        self.ax_loss.set_xlabel('Iterations')
        self.ax_loss.set_ylabel('Loss')
        
        self.ax_acc.clear()
        self.ax_acc.plot(self.acc_list)
        # ax_acc.set_title()
        self.ax_acc.set_xlabel('Iterations')
        self.ax_acc.set_ylabel('Accuracy')
        self.fig.canvas.draw()
        
        time.sleep(1)