import librosa
import numpy as np
import os
import random

GENERATE_NUM = 10000


# Load the file list of folder file-1 and the original number of files
read_path = "/mnt/nas/home/chenjiafeng/TAU dataset/files-1"
file_list = os.listdir(read_path)

path = "/mnt/nas/home/chenjiafeng/TAU dataset/files-1/"

total_num = len(file_list)

for i in range(GENERATE_NUM):
    # Randomly select the first file
    first_index = random.randint(0, total_num - 1)
    first_filename = file_list[first_index]
    
    # Obtain the label of the first file
    if (first_filename[0] == '('):
        first_filename = first_filename[6:]
    
    first_label = first_filename.split('-')[0]
    
    # Load the first file
    first_file = np.load(path + first_filename)
    
    # Randomly select the second file    
    second_index = random.randint(0, total_num - 1)
    second_filename = file_list[second_index]
        
    # Check the second label
    if (second_filename[0] == '('):
        second_filename = second_filename[6:]
        
    second_label = second_filename.split('-')[0]
    
    # Load the second file
    second_file = np.load(path + second_filename)
    
    # Generate a value of lambda from beta distribution
    param_lambda = np.random.beta(0.1, 0.9)
    
    # Mixup the two files together based on the selected lambda value
    mixed_file = param_lambda * first_file + (1 - param_lambda) * second_file
    
    # Construct the saved filename of the mixup file
    mixup_filename = "(mixup)" + str(i) + "-" + first_label + "-" + second_label + "-" + str(param_lambda)
    
    # Save the mixup file
    np.save(path + mixup_filename, mixed_file)
    print(mixup_filename + ".npy saved")