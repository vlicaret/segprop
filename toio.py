import torch
import os
import fnmatch
import numpy as np


def load(path, device='cuda'):
    ext = os.path.splitext(path)[1]
    if ext == '.npy':
        data = torch.from_numpy(np.load(path)).to(device)
    elif ext == '.npz':
        file = np.load(path)
        data = {}
        for k, v in file.items():
            if type(v) == np.ndarray:
                data[k] = torch.from_numpy(v).to(device)
            else:
                data[k] = v
    else:
        raise NotImplementedError
    return data


def load_array(path, device='cuda'):
    ext = os.path.splitext(path)[1]
    if ext == '.npy':
        data = torch.from_numpy(np.load(path))
    elif ext == '.npz':
        data = np.load(path)
        data = torch.from_numpy(next(iter(data.values())))
    else:
        raise NotImplementedError
    data = data.to(device)
    return data


def save_array(path, data, ext=None):
    if ext is None:
        ext = os.path.splitext(path)[1]
    data = data.cpu().numpy()
    if ext == '.npy':
        np.save(path, data)
    elif ext == '.npz':
        np.savez_compressed(path, data=data)
    else:
        raise NotImplementedError
    return 0


def match_files(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result
