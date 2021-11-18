import librosa
import numpy as np
import os
import random
from dataset import read_train_files


path = "/mnt/nas/home/chenjiafeng/TAU dataset/files-1/"
train_filenames = read_train_files()

for file_name in train_filenames:
    single_file = np.load(path + file_name + ".npy")
    
    crop_left = random.randint(0, 215)
    cropped_file = single_file[:, crop_left:(crop_left + 216)]
    
    left_padding_num = random.randint(0, 215)
    right_padding_num = 215 - left_padding_num
    left_padding = np.zeros((256, left_padding_num), dtype="float32")
    right_padding = np.zeros((256, right_padding_num), dtype="float32")
    
    temp_stack = np.hstack((left_padding, cropped_file))
    cropped_final = np.hstack((temp_stack, right_padding))
    
    save_file_name = "(crop)" + file_name
    
    np.save(path+save_file_name, cropped_final)
    
    print(save_file_name+".npy"+" saved.")





# Testing
'''
test_file = np.load(path + "tram-vienna-285-8639-a.npy")
crop_left = random.randint(0, 215)
cropped_file = test_file[:, crop_left:crop_left + 216]
print(crop_left)
print(cropped_file.shape)
print(cropped_file)

left_padding_num = random.randint(0, 215)
right_padding_num = 215 - left_padding_num

left_padding = np.zeros((256, left_padding_num))
right_padding = np.zeros((256, right_padding_num))

temp_stack = np.hstack((left_padding, cropped_file))
cropped_final = np.hstack((temp_stack, right_padding))

print("Left padding:", left_padding_num)
print("Right padding:", right_padding_num)
print(cropped_final.shape)
print(cropped_final)
'''