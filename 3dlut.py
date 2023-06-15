import torch
from torch import nn
import torch.nn.functional as F
from unet_model import UNet
from models_x import *

class GuideNN(nn.Module):
    def __init__(self, params=None):
        super(GuideNN, self).__init__()
        self.params = params
        self.conv1 = ConvBlock(3, 16, kernel_size=1, padding=0, batch_norm=True)
        self.conv2 = ConvBlock(16, 1, kernel_size=1, padding=0, activation= nn.Sigmoid) #nn.Tanh nn.Sigmoid

    def forward(self, x):
        return self.conv2(self.conv1(x))#.squeeze(1)

class ConvBlock(nn.Module):
    def __init__(self, inc , outc, kernel_size=3, padding=1, stride=1, use_bias=True, activation=nn.ReLU, batch_norm=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(int(inc), int(outc), kernel_size, padding=padding, stride=stride, bias=use_bias)
        self.activation = activation() if activation else None
        self.bn = nn.BatchNorm2d(outc) if batch_norm else None

        if use_bias and not batch_norm:
            self.conv.bias.data.fill_(0.00)
        # aka TF variance_scaling_initializer
        torch.nn.init.kaiming_uniform_(self.conv.weight)#, mode='fan_out',nonlinearity='relu')
        
    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x

class FC(nn.Module):
    def __init__(self, inc , outc, activation=nn.ReLU, batch_norm=False):
        super(FC, self).__init__()
        self.fc = nn.Linear(int(inc), int(outc), bias=(not batch_norm))
        self.activation = activation() if activation else None
        self.bn = nn.BatchNorm1d(outc) if batch_norm else None  
        
        if not batch_norm:
            self.fc.bias.data.fill_(0.00)
        # aka TF variance_scaling_initializer
        torch.nn.init.kaiming_uniform_(self.fc.weight)#, mode='fan_out',nonlinearity='relu')

        
    def forward(self, x):
        x = self.fc(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x
class Slice(nn.Module):
    def __init__(self):
        super(Slice, self).__init__()

    def forward(self, bilateral_grid, guidemap): 
        bilateral_grid = bilateral_grid.permute(0,3,4,2,1)
        guidemap = guidemap.squeeze(1)
        # grid: The bilateral grid with shape (gh, gw, gd, gc).
        # guide: A guide image with shape (h, w). Values must be in the range [0, 1].
        coeefs = bilateral_slice(bilateral_grid, guidemap).permute(0,3,1,2)
        return coeefs


class Slice(nn.Module):
    def __init__(self):
        super(Slice, self).__init__()

    def forward(self, bilateral_grid, guidemap):
        N, _, H, W = guidemap.shape
        hg, wg = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])
        hg = hg.type(torch.FloatTensor).repeat(N, 1, 1).unsqueeze(3) / (H-1) * 2 - 1
        wg = wg.type(torch.FloatTensor).repeat(N, 1, 1).unsqueeze(3) / (W-1) * 2 - 1
        guidemap = guidemap.permute(0,2,3,1).contiguous()
        guidemap_guide = torch.cat([guidemap, hg, wg], dim=3).unsqueeze(1)

        coeff = F.grid_sample(bilateral_grid, guidemap_guide)

        return coeff.squeeze(2)


class Coeffs(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = UNet(n_channels=9)

        
        
    def forward(self, x1, x2, x3):
        inputs = F.interpolate(torch.cat((x1, x2, x3), dim=1), size=[128, 128],   mode='bilinear', align_corners=True)
        coeff = self.net(inputs)
        return  coeff.reshape(-1, 12, 16, 16, 16)    



class StudentNet(nn.Module):

    def __init__(self):
        super(StudentNet, self).__init__()
        self.coeffs = Coeffs()
        self.guide = GuideNN()
        self.slice = Slice()


    def forward(self, low, med, high):
        coeffs = self.coeffs(low, med, high)
        guide = self.guide(0.33 *low + 0.33 * med + 0.33 * high)
        guide = F.interpolate(guide, size=[64, 64],   mode='bilinear', align_corners=True)
        slice_coeffs = self.slice(coeffs, guide).reshape(-1, 3*4*4, 16, 16)
        
        trilinear_ = TrilinearInterpolation() 
        
        result = trilinear_(slice_coeffs, 0.33 *low + 0.33 * med + 0.33 * high)
        return result







net = StudentNet()

data = torch.zeros(1, 3, 3840, 2160)

print(net(data,data,data))
