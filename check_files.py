import os 
import csv
import numpy as np
import librosa

train_filenames = []
with open("/mnt/nas/home/chenjiafeng/fold1_test.csv", 'r') as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader, 0):
        if i == 0:
            continue

        row_left = row[0].split('.')[0]
        file_name = row_left[6:]
        train_filenames.append(file_name)
  
for single_file in train_filenames:
    a = np.load("/mnt/nas/home/chenjiafeng/TAU dataset/files-1/"+single_file+".npy")


print("All files found.")