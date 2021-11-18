import librosa
import torch

file_path = "D://TAU dataset//TAU dataset//TAU-urban-acoustic-scenes-2020-mobile-development//audio//"
file_name = "airport-barcelona-0-0-a.wav"

a, sr = librosa.load(file_path+file_name, sr=None)

#print("Signal:", a)
#print("sr:", sr)

melspec = librosa.feature.melspectrogram(y=a, sr=sr, n_fft=2048, hop_length=1024, n_mels=256)

log_melspec = librosa.power_to_db(melspec)

print("Log Mel Spectrogram shape:", log_melspec.shape)

delta_logmel = librosa.feature.delta(log_melspec)
delta2_logmel = librosa.feature.delta(log_melspec, order=2)

#print("Log Mel delta shape:", delta_logmel.shape)

logmelspec_tensor = torch.from_numpy(log_melspec)
delta_tensor = torch.from_numpy(delta_logmel)
delta2_tensor = torch.from_numpy(delta2_logmel)

input_tensor = torch.stack((logmelspec_tensor, delta_tensor, delta2_tensor), dim=0)

print(input_tensor.shape)





'''
delta_delta_logmel = librosa.feature.delta(log_melspec, order=2)

print("Log Mel delta-delta shape:", delta_delta_logmel.shape)
'''


'''
mfcc = librosa.feature.mfcc(y=a, sr=sr)

print("MFCC shape:", mfcc.shape)

delta_mfcc = librosa.feature.delta(mfcc)

print("MFCC delta shape:", delta_mfcc.shape)
'''