import torch
import torch.nn as nn
import numpy as np
from dataset import TestDataset
from model import TridentModel
from torch.utils.data import DataLoader

# --------------------------- Preparation ---------------------------
BATCH_SIZE = 1
MODEL_NAME = "train_adam_net_params_1_21_1"
SAVE_FILE = "adam_test_stats_10epoch"

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

test_dataset = TestDataset()
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


if __name__ == '__main__':
    # Load the trained model
    model = TridentModel()
    model.load_state_dict(torch.load(MODEL_NAME + '.pkl'))
    model.to(device)
    
    model.eval()
    
    total_cases = 0
    correct = 0
    
    # Create a list for correctness statistics
    stats = []
    for i in range(10):
        stats.append([i, 0, 0, 0])
    
    total_amount = len(test_dataset)

    for i, (single_input, label) in enumerate(test_dataloader, 1):
        single_input = single_input.to(device)
        
        label = label.to(device)
        
        net_output = model(single_input)
        
        label_int = int(label.squeeze().item())
            
        # Delete This
        '''
        if i <= 50:
            print("Predict #", i, ": ", net_output)
            print("Target #", i, ": ", label_int)
        '''
            
        max_val, max_id = net_output.max(1)
        predict = max_id.item()
        
        if predict == label_int:
            correct += 1
            stats[label_int][2] += 1
        
        total_cases += 1
        stats[label_int][1] += 1
        
        if i % 100 == 0:
            print("Finished: " + str(i) + '/' + str(total_amount))
    
    # Calculate the accuracy for each scene
    for one_label in stats:
        one_label[3] = one_label[2] / one_label[1]
    
    # Calculate the overall accuracy
    total_stats = [-1, total_cases, correct, correct / total_cases]
    stats.append(total_stats)
    
    stats_arr = np.array(stats)
    np.savetxt(SAVE_FILE + ".csv", stats_arr, delimiter=',')
    
    print('Total Accuracy: ', correct/total_cases)
    # print('CV True Positive Accuracy: ', true_positive/true_cases)