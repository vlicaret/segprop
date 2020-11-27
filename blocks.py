import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import util


class GaussBlur3d(nn.Module):
    def __init__(self, kernel_size=5, sigma=2, pad=None, device='cuda'):
        super(GaussBlur3d, self).__init__()
        try:
            kernel_size[0]
        except TypeError:
            kernel_size = (kernel_size, kernel_size, kernel_size)
        kernel = util.gaussian(kernel_size, sigma, device=device)
        kernel = kernel.view(1, 1, kernel_size[0], kernel_size[1], kernel_size[2])
        if pad is None:
            pad = util.pad(kernel_size)
        self.pad = pad[::-1]
        self.weights = kernel

    def forward(self, mp):
        mp = F.pad(mp, self.pad, mode='replicate')   # switch to reflect once implemented
        mp = F.conv3d(mp, self.weights.expand(mp.shape[1], -1, -1, -1, -1), groups=mp.shape[1])
        return mp


class MedianFilter3d(nn.Module):
    def __init__(self, size=5, pad=None, device='cuda'):
        super(MedianFilter3d, self).__init__()
        try:
            size[0]
        except TypeError:
            size = (size, size, size)
        kernel = torch.zeros((size[0] * size[1] * size[2], size[0], size[1], size[2]), device=device)
        for i in range(size[0]):
            for j in range(size[1]):
                for k in range(size[2]):
                    kernel[i + j * size[0] + k * size[0] * size[1], i, j, k] = 1
        if pad is None:
            pad = util.pad(size)
        self.pad = pad[::-1]
        self.weights = kernel.unsqueeze(1)

    def forward(self, mp):
        mp = F.pad(mp, self.pad, mode='replicate')
        # mp = F.conv3d(mp, self.weights)
        # mp = mp.median(dim=1).values.unsqueeze(1)
        mp = F.conv3d(mp, self.weights.repeat(mp.shape[1], 1, 1, 1, 1), groups=mp.shape[1])
        mps = []
        for m in mp.split(self.weights.shape[0], dim=1):
            mps.append(m.median(dim=1).values)
        mp = torch.cat(mps, dim=1).unsqueeze(1)
        return mp


class MedianFilterWeighted3d(nn.Module):
    def __init__(self, size=5, pad=None, device='cuda'):
        super(MedianFilterWeighted3d, self).__init__()
        try:
            size[0]
        except TypeError:
            size = (size, size, size)
        kernel = torch.ones((size[0], size[1], size[2]), device=device) / (size[0] * size[1] * size[2])
        kernel = kernel.view(1, 1, size[0], size[1], size[2])
        if pad is None:
            pad = util.pad(size)
        self.pad = pad[::-1]
        self.weights = kernel

    def forward(self, mp):
        mp = F.pad(mp, self.pad, mode='replicate')
        mp = F.conv3d(mp, self.weights.expand(mp.shape[1], -1, -1, -1, -1), groups=mp.shape[1])
        mp = mp.permute(1, 0, 2, 3, 4)
        mp = mp / mp.sum(dim=0)
        med = torch.zeros((2, mp.shape[1], mp.shape[2], mp.shape[3], mp.shape[4]), device=mp.device, dtype=torch.long)
        for i in range(mp.shape[0]):
            changed = mp[i] > 0
            s = mp[:i+1].sum(dim=0)
            med[0][changed & (s < 0.5)] = i
            med[1][(med[1] == 0) & (s > 0.5)] = i
        for i in range(mp.shape[0]):
            delete = ~((med[0] == i) | (med[1] == i))
            mp[i, delete] = 0
        mp = mp.permute(1, 0, 2, 3, 4)
        return mp
