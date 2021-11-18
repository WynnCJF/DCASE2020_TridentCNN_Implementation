import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import csv

PATH = "train_loss_1_20+//"
CSVNAME = "adam_epoch"
CSVNUM = 25
FIGURENAME = "adam_loss_1_20+(25 epoch)"
NUMPEREPOCH = 44000


x = []
loss = []

for i in range(CSVNUM):
    data = []
    
    with open(PATH + CSVNAME + str(i+1) + '.csv', 'r') as f:
        reader = csv.reader(f)
    
        for row in reader:
            data.append(row)
            
    for train_num in data[0]: 
        x.append((float(train_num) + i*NUMPEREPOCH))
    
    for single_loss in data[1]:
        loss.append(float(single_loss))

plt.figure()
plt.plot(x, loss, c = 'red', linewidth = 0.5)
plt.xlabel('Training')
plt.ylabel('Loss')

plt.savefig(PATH + FIGURENAME + ".png")
