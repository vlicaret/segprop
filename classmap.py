import torch


def logical(indexed, no_classes, device=None):
    if device is None:
        device = indexed.device
    noy, nox = indexed.shape
    mapped = torch.zeros((noy, nox, no_classes), device=device, dtype=torch.uint8)
    for i in range(no_classes):
        mapped[..., i] = indexed == i
    return mapped


def vote(votes, no_classes):
    voted = torch.argmax(votes, 2)
    mp = logical(voted, no_classes)
    return mp


# TODO default colors, vectorize
def colorify(mp, colors):
    noy, nox, noc = mp.shape
    img = torch.zeros((noy, nox, 3), device=mp.device, dtype=torch.uint8)
    for i in range(noc):
        for c in range(3):
            img[:, :, c] += mp[:, :, i] * colors[i][c]
    return img


def from_colors(img, colors):
    noy, nox = img.shape[:2]
    noc = len(colors)
    mp = torch.zeros((noy, nox, noc), device=img.device, dtype=torch.bool)
    for i in range(noc):
        mp[..., i] = torch.all(img == torch.tensor(colors[i])[None, None, :], dim=2)
    return mp


def stats(mp, gt, determinate=True):
    noy, nox, noc = mp.shape
    mp = mp.type(torch.uint8)
    gt = gt.type(torch.uint8)

    merge = mp + 2 * gt

    n, p, tn, fp, fn, tp = torch.zeros(noc, device=merge.device), torch.zeros(noc, device=merge.device), torch.zeros(noc, device=merge.device),\
                           torch.zeros(noc, device=merge.device), torch.zeros(noc, device=merge.device), torch.zeros(noc, device=merge.device)
    for c in range(noc):
        n[c] = torch.sum(gt[:, :, c] == 0)
        p[c] = torch.sum(gt[:, :, c] == 1)
        layer = merge[:, :, c]
        tn[c] = torch.sum(layer == 0)
        fp[c] = torch.sum(layer == 1)
        fn[c] = torch.sum(layer == 2)
        tp[c] = torch.sum(layer == 3)

    pre = tp / (tp + fp)
    rec = tp / (tp + fn)

    if determinate:
        mistakes_pre = (p != 0) & (tp == 0) & (fp == 0)     # true gt but no representation in prediction
        mistakes_rec = (p == 0) & (fp != 0)                 # no true gt but mistakes in prediction
        # corrects_pre = (p == 0) & (fp == 0)                 # no true gt and no mistakes in prediction
        # corrects_rec = (p == 0) & (fn == 0)                 # no true gt and no mistakes in prediction
        mistakes = mistakes_rec | mistakes_pre
        pre[mistakes_pre] = 0
        rec[mistakes_rec] = 0
        # pre[corrects_pre] = 1
        # rec[corrects_rec] = 1

    fms = 2 * pre * rec / (pre + rec)

    if determinate:
        fms[mistakes] = 0

    return {
        'precision': pre.cpu(),
        'recall': rec.cpu(),
        'fmeasure': fms.cpu(),
        'p': p.cpu(),
        'n': n.cpu(),
        'tn': tn.cpu(),
        'fp': fp.cpu(),
        'fn': fn.cpu(),
        'tp': tp.cpu(),
        'mean': {
            'precision': pre[~torch.isnan(pre)].mean().cpu(),
            'recall': rec[~torch.isnan(rec)].mean().cpu(),
            'fmeasure': fms[~torch.isnan(fms)].mean().cpu(),
        }
    }
