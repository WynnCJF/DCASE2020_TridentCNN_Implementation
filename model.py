import torch
import torch.nn as nn
import torch.nn.functional as F


''' ------------------------ Basic Residual Block ------------------------ '''
class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, inter_planes, stride_first=1, dilation_first=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, affine=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=3, stride=(1, stride_first), padding=1)
        if dilation_first != 1:
            padding_h = ((dilation_first - 1) * 2 + 3) // 2
            self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=3, stride=(1, stride_first), 
                                   padding=(padding_h, 1), dilation=(dilation_first, 1))
        
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inter_planes, planes, kernel_size=3, stride=1, padding=1, dilation=1)
        
        self.identity = nn.Sequential()
        
        # The identity path construction still needs to be confirmed
        if stride_first != 1 or in_planes != planes:
            self.identity = nn.Sequential(
                nn.BatchNorm2d(in_planes),
                nn.Conv2d(in_planes, planes, kernel_size=1, stride = (1, stride_first))
            )
            
    def forward(self, x):
        out = self.relu1(self.bn1(x))
        out = self.conv1(out)
        out = self.relu2(self.bn2(out))
        out = self.conv2(out)
        
        out = out + self.identity(x)
        
        return out


''' ------------------------ Resnet Block ------------------------ '''
class ResnetBlock(nn.Module):
    def __init__(self, in_channels, inter_channels_1):
        super().__init__()
        
        self.inter_channels_1 = inter_channels_1
        
        self.bn1 = nn.BatchNorm2d(in_channels, affine=True)
        self.conv1 = nn.Conv2d(in_channels, inter_channels_1, kernel_size=3, stride=(1, 2), padding=1)
        
        self.residual_blocks = self.make_residual_blocks()
        
    def make_residual_blocks(self):
        residual_block_seq = []
        
        # Residual block 0
        residual_block_seq.append(ResidualBlock(self.inter_channels_1, self.inter_channels_1, 64))
        
        # Residual block 1-9
        for i in range(3):
            residual_block_seq.append(ResidualBlock(self.inter_channels_1, self.inter_channels_1, 64, dilation_first=2))
            residual_block_seq.append(ResidualBlock(self.inter_channels_1, self.inter_channels_1, 64, dilation_first=2))
            residual_block_seq.append(ResidualBlock(self.inter_channels_1, self.inter_channels_1, 64, stride_first=2))
        
        # Residual block 10-11
        residual_block_seq.append(ResidualBlock(self.inter_channels_1, self.inter_channels_1, 64, dilation_first=2))
        residual_block_seq.append(ResidualBlock(self.inter_channels_1, self.inter_channels_1, 64, dilation_first=2))
        
        return nn.Sequential(*residual_block_seq)
    
    def forward(self, x):
        out = self.bn1(x)
        out = self.conv1(out)
        
        out = self.residual_blocks(out)
        
        return out


''' ------------------------ Trident Model ------------------------ '''
class TridentModel(nn.Module):
    def __init__(self, inter_channels_1=256, inter_channels_2=768, inter_channels_3=10):
        super().__init__()
        
        self.resnet1 = ResnetBlock(in_channels=3, inter_channels_1=inter_channels_1)
        self.resnet2 = ResnetBlock(in_channels=3, inter_channels_1=inter_channels_1)
        self.resnet3 = ResnetBlock(in_channels=3, inter_channels_1=inter_channels_1)
        
        self.bn1 = nn.BatchNorm2d(inter_channels_1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inter_channels_1, inter_channels_2, kernel_size=1)
        
        self.bn2 = nn.BatchNorm2d(inter_channels_2)
        self.conv2 = nn.Conv2d(inter_channels_2, inter_channels_3, kernel_size=1)
        
        self.bn3 = nn.BatchNorm2d(inter_channels_3)
        self.globalaveragepool = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x_low, x_medium, x_high = torch.split(x, (64, 64, 128), dim=2)

        x_low_out = self.resnet1(x_low)
        x_medium_out = self.resnet2(x_medium)
        x_high_out = self.resnet3(x_high)
        out = torch.cat((x_low_out, x_medium_out, x_high_out), dim=2)
        
        out = self.relu1(self.bn1(out))
        out = self.conv1(out)
        
        out = self.bn2(out)
        out = self.conv2(out)
        
        out = self.bn3(out)
        out = self.globalaveragepool(out)
        out = torch.squeeze(out, dim=3)
        out = torch.squeeze(out, dim=2)
        
        return out    