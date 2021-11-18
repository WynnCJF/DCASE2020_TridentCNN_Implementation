import torch
import torch.nn as nn

a = torch.randn(1, 1, 64, 53)

conv_layer = torch.nn.Conv2d(1, 1, kernel_size=(3, 3), padding=(2, 1), dilation=(2, 1))

b = conv_layer(a)

print(a.shape)
print(b.shape)

c_original = torch.randn(1, 3, 256, 431)
c = torch.split(c_original, (64, 64, 128), dim=2)

print(c[0].shape, c[1].shape, c[2].shape)

c_cat = torch.cat((c[0], c[1], c[2]), dim=2)
print(c_cat.shape)

d = torch.randn(1, 1, 64, 53)
globalaveragepool = nn.AdaptiveAvgPool2d(1)
d_out = globalaveragepool(d)
print(d_out.shape)