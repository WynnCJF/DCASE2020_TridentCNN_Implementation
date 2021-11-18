import librosa
import librosa.display
import torch
import numpy as np
from matplotlib import pyplot as plt

a = librosa.tone(1000, sr=44100, duration=10)
spec = librosa.stft(a, n_fft=2048, hop_length=1024)
spec_scale = np.abs(spec)**2

print(a.shape)
print(spec.shape)

'''
plt.figure()
librosa.display.specshow(np.abs(spec)**2,
                            sr=44100,
                            hop_length=1024,
                            x_axis="time",
                            y_axis="linear")
plt.colorbar(format="%+2.f")
plt.show()
'''

for freq, mag in enumerate(spec_scale, 0):
    if np.mean(mag) != 0:
        print(np.mean(mag))
        print(freq)