import librosa
import numpy as np
import os

# Read in subsequent .wav files

path = "/mnt/nas/home/chenjiafeng/TAU dataset/TAU-urban-acoustic-scenes-2020-mobile-development-1/audio"
path_list = os.listdir(path)

save_path = "/mnt/nas/home/chenjiafeng/TAU dataset/files-1/"

for file_name in path_list:
    cropped_wavname = file_name[:-4]
    
    cur_file, sr = librosa.load(path+"/"+file_name, sr=None)
    melspec = librosa.feature.melspectrogram(y=cur_file, sr=sr, n_fft=2048, hop_length=1024, n_mels=256)
    
    np.save(save_path+cropped_wavname, melspec)
    
    print(cropped_wavname+".npy"+" saved.")
    

'''
# Test
file_path = "D://MIR//DCASE//Files//"
file_name = "tram-vienna-285-8639-a.wav"

cropped_filename = file_name[:-4]

#print(cropped_filename)

a, sr = librosa.load(file_path+file_name, sr=None)

melspec = librosa.feature.melspectrogram(y=a, sr=sr, n_fft=2048, hop_length=1024, n_mels=256)

print(melspec)

np.save(file_path+cropped_filename, melspec)

loaded_melspec = np.load(file_path+cropped_filename+".npy")
print(loaded_melspec)

np.save(file_path+cropped_filename+"1", loaded_melspec)
np.savetxt(file_path+cropped_filename+"2.csv", loaded_melspec, delimiter=',')
'''