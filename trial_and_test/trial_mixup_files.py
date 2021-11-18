import librosa
import numpy as np
import os


file_1 = np.load("airport-barcelona-0-0-a.npy")
file_2 = np.load("airport-barcelona-0-1-a.npy")

print(file_1)
print(file_2)

beta_trial = np.random.beta(0.1, 0.9)
print("lambda =", beta_trial)

file_sum = beta_trial * file_1 + (1 - beta_trial) * file_2
print(file_sum)

print("Type before sum:", type(file_1[0][0]))
print("Type after sum:", type(file_sum[0][0]))