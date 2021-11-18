import os
import csv
import numpy as np
import torch
import librosa
from torch.utils.data import Dataset
import enum

# Label
label_search = {
    "airport": 0,
    "shopping_mall": 1,
    "metro_station": 2,
    "street_pedestrian": 3,
    "public_square": 4,
    "street_traffic": 5,
    "tram": 6,
    "bus": 7,
    "metro": 8,
    "park": 9,
}

def read_train_files():
    """
    Return a list of all filenames in the selected train set
    """
    
    train_filenames = []
    with open("/mnt/nas/home/chenjiafeng/fold1_train.csv", 'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader, 0):
            # Skip the title
            if i == 0:
                continue
        
            # Discard the prefix and "audio/" in the front
            row_left = row[0].split('.')[0]
            file_name = row_left[6:]
            train_filenames.append(file_name)
    
    return train_filenames

def read_train_files_from_dir():
    """
    Return a list of all filenames in the training files folder
    """
    
    read_path = "/mnt/nas/home/chenjiafeng/TAU dataset/files-1"
    train_file_list = os.listdir(read_path)
    
    return train_file_list

def read_test_files():
    """
    Return a list of all filenames in the selected test set
    """
    
    test_filenames = []
    with open("/mnt/nas/home/chenjiafeng/fold1_test.csv", 'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader, 0):
            # Skip the title
            if i == 0:
                continue
        
            # Discard the prefix and "audio/" in the front
            row_left = row[0].split('.')[0]
            file_name = row_left[6:]
            test_filenames.append(file_name)
    
    return test_filenames
    

def obtain_label(file_name):
    """
    Return the corresponding label int given a filename
    """
    place = file_name.split("-")[0]
    
    label_int = 0
    
    try:
        label_int = int(label_search[place])
    except:
        print("Label " + place + " not found")
    
    return label_int
    

''' -------------------------- Dataset -------------------------- '''
class TrainDataset(Dataset):
    def __init__(self):
        self.training_filenames = read_train_files_from_dir()
        self.files_num = len(self.training_filenames)
    
    def __getitem__(self, index):
        npy_file_name = self.training_filenames[index]
        
        # Load the Mel spectrogram
        melspec = np.load("/mnt/nas/home/chenjiafeng/TAU dataset/files-1/" + npy_file_name)
        
        # Convert the scale to db
        log_melspec = librosa.power_to_db(melspec)

        # Calculate delta and delta-delta
        delta_logmel = librosa.feature.delta(log_melspec)
        delta2_logmel = librosa.feature.delta(log_melspec, order=2)

        # Stack the logmel spectrogram, delta, and delta-delta tensor
        logmelspec_tensor = torch.from_numpy(log_melspec)
        delta_tensor = torch.from_numpy(delta_logmel)
        delta2_tensor = torch.from_numpy(delta2_logmel)

        input_tensor = torch.stack((logmelspec_tensor, delta_tensor, delta2_tensor), dim=0)
        
        # Identify and create the corresponding label
        label = np.zeros(10, dtype="float32")
        
        # Check whether the file is a mixup or crop, and do the corresponding change to the label vector
        if npy_file_name[0] == '(':
            if npy_file_name[1] == 'c':
                npy_file_name = npy_file_name[6:]
                label_int = obtain_label(npy_file_name)
                
                label[label_int] = 1
            else:
                split_list = npy_file_name.split('-')
                
                first_label_int = int(label_search[split_list[1]])
                second_label_int = int(label_search[split_list[2]])
                
                if len(split_list) == 4:
                    param_lambda_str = split_list[3][:-4]
                else:
                    param_lambda_str = split_list[3] + "-" + split_list[4][:-4]
                    
                param_lambda = float(param_lambda_str)
                
                label[first_label_int] = param_lambda
                label[second_label_int] = 1 - param_lambda
        else:
            label_int = obtain_label(npy_file_name)
            label[label_int] = 1

        label_out = torch.from_numpy(label)
        
        return input_tensor, label_out
        
    def __len__(self):
        return self.files_num
    
    def check_filename(self, index):
        return self.training_filenames[index]


class TestDataset(Dataset):
    def __init__(self):
        self.test_filenames = read_test_files()
        self.files_num = len(self.test_filenames)
    
    def __getitem__(self, index):
        npy_file_name = self.test_filenames[index]
        melspec = np.load("/mnt/nas/home/chenjiafeng/TAU dataset/files-2/" + npy_file_name + ".npy")
        
        # Convert the scale to db
        log_melspec = librosa.power_to_db(melspec)

        # Calculate delta and delta-delta
        delta_logmel = librosa.feature.delta(log_melspec)
        delta2_logmel = librosa.feature.delta(log_melspec, order=2)

        # Stack the logmel spectrogram, delta, and delta-delta tensor
        logmelspec_tensor = torch.from_numpy(log_melspec)
        delta_tensor = torch.from_numpy(delta_logmel)
        delta2_tensor = torch.from_numpy(delta2_logmel)

        input_tensor = torch.stack((logmelspec_tensor, delta_tensor, delta2_tensor), dim=0)
        
        # Identify and create the corresponding label
        label_int = obtain_label(npy_file_name)
        
        label = torch.Tensor([label_int])
        
        return input_tensor, label
        
    def __len__(self):
        return self.files_num