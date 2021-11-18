import numpy as np
import os
import random
from dataset import read_test_files

path = "/mnt/nas/home/chenjiafeng/TAU dataset/files-1/"
save_path = "/mnt/nas/home/chenjiafeng/TAU dataset/files-2/"

test_filenames = read_test_files()

for single_filename in test_filenames:
    # Move the test files to files-2 folder (done)
    '''
    # Read a test file
    read_file = np.load(path + single_filename + ".npy")
    
    # Directly save the file to files-2 folder
    np.save(save_path + single_filename, read_file)
    
    print(single_filename + ".npy moved.")
    '''
    
    # Delete the test files from files-1 folder
    os.remove(path + single_filename + ".npy")
    
    print(single_filename + ".npy successfully removed.")