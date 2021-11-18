import torch
import torch.nn as nn
import torch.nn.functional as F
from model import ResidualBlock, ResnetBlock, TridentModel
import numpy as np
import librosa

# Test basic residual block
'''
a = torch.randn(1, 3, 256, 431)
residualblock = ResidualBlock(3, 256, 64, 2, 2)
a_out = residualblock(a)

print(a_out.shape)
'''

# Test resnet block
'''
b = torch.randn(1, 3, 256, 431)
resnetblock = ResnetBlock(in_channels=3, inter_channels_1=256)
b_out = resnetblock(b)

print(b_out.shape)
'''

# Test trident model
'''
c = torch.randn(1, 3, 256, 431)
trident = TridentModel()
c_out = trident(c)

print(c_out)
'''

'''
melspec = np.load("tram-vienna-285-8639-a.npy")
log_melspec = librosa.power_to_db(melspec)
print(log_melspec.shape)

delta_logmel = librosa.feature.delta(log_melspec)
delta2_logmel = librosa.feature.delta(log_melspec, order=2)

logmelspec_tensor = torch.from_numpy(log_melspec)
delta_tensor = torch.from_numpy(delta_logmel)
delta2_tensor = torch.from_numpy(delta2_logmel)

d = torch.stack((logmelspec_tensor, delta_tensor, delta2_tensor), dim=0)
d = d.unsqueeze(0)
print(d.shape)

trident = TridentModel()
d_out = trident(d)
d_out_array = d_out.detach().numpy()

print(d_out_array)
'''

'''
e = torch.Tensor([[ 4.9671,  1.1590,  1.0210, -2.7595, -1.4073,
          0.0000, -2.8974, -1.5177,  3.3114,  1.4625]])
softmax = nn.Softmax(dim=1)
e_out = softmax(e)

print(e_out)
'''

conv1 = nn.Conv2d(3, 256, kernel_size=1)
print(conv1.__class__.__name__)