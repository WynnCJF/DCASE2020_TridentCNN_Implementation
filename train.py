import torch
import torch.nn as nn
import numpy as np
from dataset import TrainDataset
from model import TridentModel
from focal_loss import FocalLoss
from focal_loss_test import MultiFocalLoss
from torch.utils.data import DataLoader

# --------------------------- Preparation ---------------------------
BATCH_SIZE = 4
EPOCH = 30
LR = 0.00001
RECORD = 100

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_dataset = TrainDataset()
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

print("Training dataset loaded. Data size:", len(train_dataset))

# Delete This
# print(train_dataset[0])
# print(train_dataset[0][0].shape)


# --------------------------- Training function ---------------------------

def train():
    running_loss = 0.0
    losses = []
    batches = []
    
    for i, (single_input, label) in enumerate(train_dataloader, 1):
        single_input = single_input.to(device)

        label = label.to(device)
        
        net_output = model(single_input)
        
        optimizer.zero_grad()
        
        label = label.squeeze()
        
        loss = criterion(net_output, label.long())
        
        loss.backward()
        optimizer.step()
        
        running_loss = running_loss + loss.item()
        
        if i%RECORD == 0:
            print('[' + str(i*BATCH_SIZE) + '/' + str(len(train_dataset)) + "]")
            loss_stat = running_loss / (BATCH_SIZE * RECORD)
            running_loss = 0.0
            print('Loss: ' + str(loss_stat))
            losses.append(loss_stat)
            batches.append(i*BATCH_SIZE)
        
    return batches, losses

#*********************************Main Cycle*********************************#
if __name__ == '__main__':
    model = TridentModel()
    model.load_state_dict(torch.load('train_net_params_1_10.pkl'))
    model.to(device)
    
    model.train()
    
    criterion = MultiFocalLoss(num_class=10)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    
    for epoch in range(1,EPOCH+1):
        print("Epoch " + str(epoch + 90) + " starts: ---------------------------------------------------")
        (train_batches,train_losses) = train()
        epoch_stats = [train_batches,train_losses]
        epoch_stats_arr = np.array(epoch_stats)
        np.savetxt('pretrain_epoch' + str(epoch + 90) + '.csv',epoch_stats_arr, delimiter=',', fmt='%s')
    
    #torch.save(model,'net.pkl')
    torch.save(model.state_dict(), 'train_net_params_1_10_2.pkl')