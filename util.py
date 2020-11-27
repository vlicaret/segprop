import torch
import math


def bound(values, minimum, maximum):
    return torch.max(torch.min(values, torch.tensor(maximum, device=values.device, dtype=values.dtype)),
                     torch.tensor(minimum, device=values.device, dtype=values.dtype))


def nan(shape, device=None, dtype=None):
    return torch.zeros(shape, device=device, dtype=dtype) * torch.tensor(float('nan'), device=device, dtype=dtype)


def pad(size):
    size = [i + 1 if i < 1 else i for i in size]
    return [math.ceil(size[i // 2] / 2) - 1 if i % 2 else math.floor(size[i // 2] / 2) for i in range(2 * len(size))]


# TODO
def epsilon(dtype=torch.float, device='cuda'):
    return torch.tensor(1e-16, device=device, dtype=dtype)


def gaussian(kernel_size, sigma, device='cuda', normalize=True):
    try:
        kernel_size[0]
    except TypeError:
        kernel_size = [kernel_size]
    try:
        sigma[0]
    except TypeError:
        sigma = [sigma]
    for i in range(len(kernel_size)):
        try:
            sigma[i]
        except IndexError:
            sigma.append(sigma[0])
    nod = len(kernel_size)
    vects = [torch.arange(k, dtype=torch.float, device=device) for k in kernel_size]
    grid = torch.stack(torch.meshgrid(vects), dim=-1)
    mean = torch.Tensor([(k - 1) / 2. for k in kernel_size]).to(device)
    sigma = torch.Tensor(sigma).to(device)
    kernel = (1 / (math.sqrt(2 * math.pi) ** nod * torch.prod(sigma))) * torch.exp(-torch.sum((grid - mean) ** 2. / (2 * sigma ** 2), dim=-1))
    if normalize:
        kernel = kernel / torch.sum(kernel)
    return kernel


def gradient(field, dim=None):
    if dim is None:
        dim = torch.arange(field.dim())
    try:
        dim[0]
    except TypeError:
        dim = [dim]
    grad = []
    for d in dim:
        sls = [slice(0, None) if i != d else slice(1, None) for i in range(field.dim())]
        sle = [slice(0, None) if i != d else slice(-1) for i in range(field.dim())]
        kps = [slice(0, None) if i != d else slice(0, 1) for i in range(field.dim())]
        kpe = [slice(0, None) if i != d else slice(-1, None) for i in range(field.dim())]
        g1 = field[sls] - field[sle]
        g2 = (g1[sls] + g1[sle]) / 2
        grad.append(torch.cat((g1[kps], g2, g1[kpe]), dim=d))
    if len(dim) == 1:
        grad = grad[0]
    return grad


def divergence(field):
    vecdim = field.dim() - 1
    div = torch.zeros_like(field[..., 0])
    for d in range(field.shape[vecdim]):
        div += gradient(field[..., d], dim=d)
    return div


def interp2d(vff, vcf, vfc, vcc, blendy, blendx):
    newdims = len(vff.shape) - len(blendy.shape)
    for i in range(newdims):
        blendy.unsqueeze_(-1)
        blendx.unsqueeze_(-1)
    dff = (1 - blendy) * (1 - blendx)
    dcf = blendy * (1 - blendx)
    dfc = (1 - blendy) * blendx
    dcc = blendy * blendx
    return dff * vff.type(blendy.dtype) + dcf * vcf.type(blendy.dtype) + dfc * vfc.type(blendy.dtype) + dcc * vcc.type(blendy.dtype)


def find_nearest(value, array, number=None):
    array = torch.abs(torch.tensor(array) - value)
    if number is None:
        idx = array.argmin()
        return idx
    else:
        inds = torch.argsort(array)
        # sl = [slice(0, None) if i != dim else slice(0, number) for i in range(array.dim())]
        return inds[:number]
