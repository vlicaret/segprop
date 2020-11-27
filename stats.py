import torch
import classmap
import toio
import os
import numpy as np
import cv2


def compare_set(map_path, gt_path, device='cuda', determinate=True):
    gt_maps = sorted(os.listdir(gt_path))

    stats = []
    for g in gt_maps:
        # verbose
        # print(g)

        # load data
        gt = toio.load_array(os.path.join(gt_path, g), device=device)
        try:
            m = toio.match_files(os.path.splitext(g)[0] + '*', map_path)[0]
            mp = toio.load_array(os.path.join(map_path, m), device=device)
        except (FileNotFoundError, IndexError):
            # print(g + ' not found, skipping.')
            continue

        # calculate stats
        stats.append(classmap.stats(mp, gt))

    noy, nox, noc = gt.shape
    nos = len(stats)

    # aggregate data
    n, p, tn, fp, fn, tp = torch.zeros(noc, dtype=torch.double), torch.zeros(noc), torch.zeros(noc),\
                           torch.zeros(noc), torch.zeros(noc), torch.zeros(noc)
    pre, rec, fms = torch.zeros((nos, noc)), torch.zeros((nos, noc)), torch.zeros((nos, noc))
    for i, s in enumerate(stats):
        n += s['n']
        p += s['p']
        tn += s['tn']
        fp += s['fp']
        fn += s['fn']
        tp += s['tp']

        pre[i, :] = s['precision']
        rec[i, :] = s['recall']
        fms[i, :] = s['fmeasure']

    pre_mi = tp / (tp + fp)
    rec_mi = tp / (tp + fn)
    
    if determinate:
        mistakes_pre = (p != 0) & (tp == 0) & (fp == 0)     # true gt but no representation in prediction
        mistakes_rec = (p == 0) & (fp != 0)                 # no true gt but mistakes in prediction
        # corrects_pre = (p == 0) & (fp == 0)                 # no true gt and no mistakes in prediction
        # corrects_rec = (p == 0) & (fn == 0)                 # no true gt and no mistakes in prediction
        mistakes = mistakes_rec | mistakes_pre
        pre_mi[mistakes_pre] = 0
        rec_mi[mistakes_rec] = 0
        # pre[corrects_pre] = 1
        # rec[corrects_rec] = 1
    
    fms_mi = 2 * pre_mi * rec_mi / (pre_mi + rec_mi)
    
    if determinate:
        fms_mi[mistakes] = 0

    pre_ma, rec_ma, fms_ma = torch.zeros(noc), torch.zeros(noc), torch.zeros(noc)
    for c in range(noc):
        pre_ma[c] = pre[~torch.isnan(pre[:, c]), c].mean()
        rec_ma[c] = rec[~torch.isnan(rec[:, c]), c].mean()
        fms_ma[c] = fms[~torch.isnan(fms[:, c]), c].mean()

    # verbose
    # print('done!')

    return {
        'p': p.numpy(),
        'n': n.numpy(),
        'tn': tn.numpy(),
        'fp': fp.numpy(),
        'fn': fn.numpy(),
        'tp': tp.numpy(),
        'micro': {
            'precision': pre_mi.numpy().tolist(),
            'recall': rec_mi.numpy().tolist(),
            'fmeasure': fms_mi.numpy().tolist(),
            'mean': {
                'precision': pre_mi[~torch.isnan(pre_mi)].mean().numpy().tolist(),
                'recall': rec_mi[~torch.isnan(rec_mi)].mean().numpy().tolist(),
                'fmeasure': fms_mi[~torch.isnan(fms_mi)].mean().numpy().tolist(),
            }
        },
        'macro': {
            'precision': pre_ma.numpy().tolist(),
            'recall': rec_ma.numpy().tolist(),
            'fmeasure': fms_ma.numpy().tolist(),
            'mean': {
                'precision': pre_ma[~torch.isnan(pre_ma)].mean().numpy().tolist(),
                'recall': rec_ma[~torch.isnan(rec_ma)].mean().numpy().tolist(),
                'fmeasure': fms_ma[~torch.isnan(fms_ma)].mean().numpy().tolist(),
            }
        }
    }


def evaluate(test_path, label_path):
    tests = sorted(os.listdir(test_path))

    acc = {'n': 0, 'p': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0, 'fmeasure': []}
    for test in tests:
        video = '_'.join(test.split('_')[0:2])
        # key = '_'.join(test.split('_')[4:])
        s = compare_set(os.path.join(test_path, test), os.path.join(label_path, video), device='cpu')
        
        acc['n'] += s['n']
        acc['p'] += s['p']
        acc['tn'] += s['tn']
        acc['fp'] += s['fp']
        acc['fn'] += s['fn']
        acc['tp'] += s['tp']
        acc['fmeasure'].append(s['micro']['mean']['fmeasure'])
    
    pre_mi = acc['tp'] / (acc['tp'] + acc['fp'])
    rec_mi = acc['tp'] / (acc['tp'] + acc['fn'])
    
    fmeasure = 2 * pre_mi * rec_mi / (pre_mi + rec_mi)
    fmeasure = np.nanmean(fmeasure)

    miou = acc['tp'] / (acc['tp'] + acc['fp'] + acc['fn'])
    miou = np.nanmean(miou)
    
    return fmeasure, miou


def evaluate_all(test_path, label_path):
    runs = sorted(os.listdir(test_path))

    for run in runs:
        name = '_'.join(run.split('_')[4:])
        fmeasure, miou = evaluate(os.path.join(test_path, run), label_path)

        print(run, 'fmeasure', '{:.4f}'.format(fmeasure), 'miou', '{:.4f}'.format(miou))

    return 0
