import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CELoss(nn.Module):    
    def __init__(self, size_average=True):
        super().__init__()
        self.size_average = size_average
        self.softmax = nn.Softmax(dim=1)

    def forward(self, preds, labels):
        preds = self.softmax(preds) # Checked
        print(preds)
        print(labels)
        loss = torch.sum(-labels * torch.log(preds), dim=1)
        
        print(loss)
        
        if self.size_average:        
            loss = loss.mean()        
        else:
            loss = loss.sum()
                
        return loss