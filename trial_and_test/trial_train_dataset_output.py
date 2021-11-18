import torch
import random
from dataset import TrainDataset

train_dataset = TrainDataset()

while True:
    i = random.randint(0, len(train_dataset))
    
    input_tensor, label = train_dataset[i]
    
    if (train_dataset.check_filename(i)[0:2] == "(m"):
        print(input_tensor)
        print(train_dataset.check_filename(i))
        print(label)
        
        break

while True:
    i = random.randint(0, len(train_dataset))
    
    input_tensor, label = train_dataset[i]
    
    if (train_dataset.check_filename(i)[0:2] == "(c"):
        print(input_tensor)
        print(train_dataset.check_filename(i))
        print(label)
        
        break

while True:
    i = random.randint(0, len(train_dataset))
    
    input_tensor, label = train_dataset[i]
    
    if (train_dataset.check_filename(i)[0] != '(') :
        print(input_tensor)
        print(train_dataset.check_filename(i))
        print(label)
        
        break