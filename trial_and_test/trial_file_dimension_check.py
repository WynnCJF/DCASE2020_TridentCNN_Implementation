import librosa
import numpy as np
import os

# Read in subsequent .wav files

path = "/mnt/nas/home/chenjiafeng/TAU dataset/files-1/"

single_file = np.load(path + "tram-vienna-285-8639-a.npy")

print(single_file.shape)
print(single_file)