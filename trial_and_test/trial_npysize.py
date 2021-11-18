import librosa
import numpy as np
import os
import random
from dataset import read_train_files

test_file = np.load("a.npy")
print(type(test_file))
print(type(test_file[0][0]))


crop_left = random.randint(0, 215)
cropped_file = test_file[:, crop_left:crop_left + 216]
print(crop_left)
print(cropped_file.shape)
print(cropped_file)
print(type(cropped_file[0][0]))

left_padding_num = random.randint(0, 215)
right_padding_num = 215 - left_padding_num

left_padding = np.zeros((256, left_padding_num), dtype='float32')
right_padding = np.zeros((256, right_padding_num), dtype='float32')

temp_stack = np.hstack((left_padding, cropped_file))
cropped_final = np.hstack((temp_stack, right_padding))

print("Left padding:", left_padding_num)
print("Right padding:", right_padding_num)
print(cropped_final.shape)
print(cropped_final)

print(type(cropped_final[0][0]))
