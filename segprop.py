import os
import math
import warnings
import flow
import map2d
import util
import classmap
import blocks
import toio
import torch
import h5py
import numpy as np
from shutil import copyfile
from scipy import ndimage
import cv2


def vote(flow_h5_path, gt_path, output_path, key_weight=1, key_homography_weight=0, cur_weight=1, dist_weighting_beta=1, precalc_flow=True,
         interpolate=False, vector_mult=1, frame_filter=None, start_from=None, stop_at=None, device='cuda', overwrite=True):
    """
    Generates intermediary automatic labels between consecutive ground truth segmentations.
    It requires sparse annotations (frame_000, frame_050, frame_0121  etc.) and precalculated optical flows.
    Flow data should be of the shape [no_frames, height, width, 2], with dim3 = [x_vectors, y_vectors].
    :param flow_h5_path: String Tuple. Tuple containing paths to (flow_forward.h5, flow_backward.h5).
    :param gt_path: String. Path to folder containing ground truth segmentations. A hard map of True/False values with one channel per class is expected. [name]_000NNN.npy/z etc.
    :param output_path: String. Output folder.
    :param key_weight: Float. Key frame projection weight.
    :param key_homography_weight: Float. Key frame projection by homographies weight. SLOW. Not computed if 0.
    :param cur_weight: Float. Current frame projection weight.
    :param dist_weighting_beta: Float. Frame distance weighting by an exponential function. 0 equals linear weighting.
    :param precalc_flow: Bool. Precalculate optical flow coordinates between consecutive _g_ts. _faster, but more memory is used.
    :param interpolate: Bool. Bilinearly interpolate coordinates during flow projection. _might be useful for low resolution data and long propagation distances.
    :param vector_mult: Float. Multiplies default flow vectors by the specified amount.
    :param frame_filter: Function. Receives frame number and returns 'True' for any frame that should be skipped. Useful for evaluation.
    :param start_from: Int. Start from frame number.
    :param stop_at: Int. Stop on frame number.
    :param device: pytorch device to run on.
    :param overwrite: Bool. Overwrite previous results.
    :return: 0 if completed successfully.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # meta
    gt_maps = sorted(os.listdir(gt_path))
    gt_no = [_frame_no(name) for name in gt_maps]
    gt_blocks = [(gt_no[i], gt_no[i + 1]) for i in range(len(gt_no) - 1)]
    noy, nox, noc = toio.load_array(os.path.join(gt_path, gt_maps[0]), device='cpu').shape
    basename = _basename(gt_maps[0])
    if start_from is None:
        start_from = gt_no[0]
    if stop_at is None or stop_at > gt_no[-1]:
        stop_at = gt_no[-1]

    # process blocks
    for block in range(len(gt_blocks)):
        start, end = gt_blocks[block]
        pre_no, nxt_no = start, end

        # skip loading data if nothing to do
        nothing_to_do = True
        for i in range(max(start, start_from), min(end, stop_at)):
            if frame_filter is None or (frame_filter is not None and not frame_filter(i)):
                if overwrite or (not overwrite and not os.path.exists(os.path.join(output_path, basename + '_{:06d}'.format(i) + '.npz'))):
                    nothing_to_do = False
                    break
        if nothing_to_do:
            continue

        # load flows
        flow_data_fwd = h5py.File(flow_h5_path[0], 'r')
        flow_data_fwd = flow_data_fwd['flow'][pre_no:nxt_no]
        flow_data_fwd = torch.from_numpy(flow_data_fwd).to(device).flip(3)
        flow_data_bkw = h5py.File(flow_h5_path[1], 'r')
        flow_data_bkw = flow_data_bkw['flow'][pre_no:nxt_no]
        flow_data_bkw = torch.from_numpy(flow_data_bkw).to(device).flip(0).flip(3)

        # precalc flows
        if precalc_flow:
            pre_fw_flowed = _flow_full(flow_data_fwd, vector_mult=vector_mult, interpolate=interpolate)
            nxt_bk_flowed = _flow_full(flow_data_bkw, vector_mult=vector_mult, interpolate=interpolate)

        # load maps
        pre_gt = toio.load_array(os.path.join(gt_path, gt_maps[block]), device=device)
        nxt_gt = toio.load_array(os.path.join(gt_path, gt_maps[block + 1]), device=device)

        for i in range(max(start, start_from), min(end, stop_at)):
            path = os.path.join(output_path, basename + '_{:06d}'.format(i) + '.npz')

            # only compute frames that aren't filtered
            if frame_filter is not None and frame_filter(i):
                continue

            # skip if present
            if not overwrite and os.path.exists(path):
                continue

            # verbose
            print('{:06d}'.format(i))

            # init
            vote_map = torch.zeros((noy, nox, noc), device=device, dtype=torch.float)
            vote_weight = key_weight + cur_weight + key_homography_weight

            # distance weighting
            delta = (i - start) / (end - start)
            pre_weight = math.exp(-dist_weighting_beta * delta)
            nxt_weight = math.exp(-dist_weighting_beta * (1 - delta))
            _ = pre_weight + nxt_weight
            pre_weight, nxt_weight = pre_weight / _, nxt_weight / _

            # gt towards current frame
            if precalc_flow:
                pre_fw_map, pre_fw_map_h = _end_map(pre_fw_flowed[..., i - pre_no], pre_gt, use_homography=key_homography_weight != 0)
                nxt_bk_map, nxt_bk_map_h = _end_map(nxt_bk_flowed[..., nxt_no - i], nxt_gt, use_homography=key_homography_weight != 0)
            else:
                pre_fw_map, pre_fw_map_h = _flow_end_map(flow_data_fwd[:i - pre_no], pre_gt, vector_mult=vector_mult, interpolate=interpolate, use_homography=key_homography_weight != 0)
                nxt_bk_map, nxt_bk_map_h = _flow_end_map(flow_data_bkw[:nxt_no - i], nxt_gt, vector_mult=vector_mult, interpolate=interpolate, use_homography=key_homography_weight != 0)

            # current frame towards ends
            cur_fw_map = _flow_cur_map(flow_data_fwd[i - pre_no:], nxt_gt, vector_mult=vector_mult, interpolate=interpolate)
            cur_bk_map = _flow_cur_map(flow_data_bkw[nxt_no - i:], pre_gt, vector_mult=vector_mult, interpolate=interpolate)

            # gather votes
            vote_map += (pre_fw_map * pre_weight + nxt_bk_map * nxt_weight) * key_weight + (cur_fw_map * nxt_weight + cur_bk_map * pre_weight) * cur_weight
            if key_homography_weight != 0:
                vote_map += (pre_fw_map_h * pre_weight + nxt_bk_map_h * nxt_weight) * key_homography_weight
            vote_map = vote_map / vote_weight

            # construct new map
            voted = torch.argmax(vote_map.type(torch.float), 2)
            voted = classmap.logical(voted, noc)

            # save map
            np.savez_compressed(path, map=voted.cpu().numpy().astype(bool), votes=vote_map.cpu().numpy())

    # copy gt over
    for i, gt in enumerate(gt_maps):
        if frame_filter is not None and frame_filter(gt_no[i]):
            continue
        copyfile(os.path.join(gt_path, gt), os.path.join(output_path, gt))

    # verbose
    # print('done!')

    return 0


def iterate(flow_h5_path, pv_path, output_path, pv_series=None, key_weight=1, key_homography_weight=0, cur_weight=1, pv_weight=None, interpolate=False,
            vector_mult=1, frame_filter=None, frame_copy=None, start_from=None, stop_at=None, device='cuda', overwrite=True):
    """
    Runs a segprop iteration over previously generated labels.
    :param flow_h5_path: String tuple. Tuple containing paths to (flow_forward.h5, flow_backward.h5)
    :param pv_path: String. Path to previous iteration folder.
    :param output_path: String. Output folder.
    :param pv_series: Int array. Frames to consider while updating votes. Ex [0, 5] = 0 - current frame, 5 - frames at -5,+5 distance.
    :param key_weight: Float. Key frame projection weight.
    :param key_homography_weight: Float. Key frame projection by homographies weight. SLOW. Not computed if 0.
    :param cur_weight: Float. Current frame projection weight.
    :param pv_weight: Float Array. Frame series based weighting. Defaults to linear.
    :param interpolate: Bool. Bilinearly interpolate coordinates during flow projection. Might be useful for low resolution data and long propagation distances.
    :param vector_mult: Float. Multiplies default flow vectors by the specified amount.
    :param frame_filter: Function. Receives frame number and returns 'True' for any frame that should be skipped. Useful for evaluation.
    :param frame_copy: Function. Receives frame number and returns 'True' for any frame that should be forwarded unchanged. Eg: GT
    :param start_from: Int. Start from frame number.
    :param stop_at: Int. Stop on frame number.
    :param device: py_torch device to run on.
    :param overwrite: Bool. Overwrite previous results.
    :return: 0 if completed successfully.
    """
    if pv_series is None:
        pv_series = [0]

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # meta
    pv_maps = sorted(os.listdir(pv_path))
    pv_no = [_frame_no(name) for name in pv_maps]
    noy, nox, noc = toio.load_array(os.path.join(pv_path, pv_maps[0]), device=device).shape
    basename = _basename(pv_maps[0])
    pv_series = sorted(pv_series)
    depth = pv_series[-1]
    nof = pv_no[-1]
    if pv_weight is None:
        vote_weight = key_weight + cur_weight + key_homography_weight
        pv_weight = pv_series.count(0) + 2 * vote_weight * (len(pv_series) - pv_series.count(0))
        pv_weight = [1 / pv_weight for _ in pv_series]
    if start_from is None:
        start_from = pv_no[0]
    if stop_at is None or stop_at > pv_no[-1]:
        stop_at = pv_no[-1]

    # process
    for i in range(max(depth + pv_no[0], start_from), min(nof - depth + 1, stop_at)):
        path = os.path.join(output_path, basename + '_{:06d}'.format(i) + '.npz')

        # only compute frames that aren't filtered
        if frame_filter is not None and frame_filter(i):
            continue

        # skip if present
        if not overwrite and os.path.exists(path):
            continue

        # forward gt frames
        if frame_copy is not None and frame_copy(i):
            copyfile(os.path.join(pv_path, pv_maps[i - pv_no[0]]), path)
            continue

        # verbose
        print('{:06d}'.format(i))

        # init
        vote_map = torch.zeros((noy, nox, noc), device=device, dtype=torch.float)

        # load flows
        flow_data_fwd = h5py.File(flow_h5_path[0], 'r')
        flow_data_fwd = flow_data_fwd['flow'][i - depth:i + depth]
        flow_data_fwd = torch.from_numpy(flow_data_fwd).to(device).flip(3)
        flow_data_bkw = h5py.File(flow_h5_path[1], 'r')
        flow_data_bkw = flow_data_bkw['flow'][i - depth:i + depth]
        flow_data_bkw = torch.from_numpy(flow_data_bkw).to(device).flip(0).flip(3)

        # load map
        cur_map = toio.load(os.path.join(pv_path, pv_maps[i - pv_no[0]]), device=device)['votes'].type(torch.float)
        for j, k in enumerate(pv_series):
            # add current votes
            if k == 0:
                vote_map += cur_map * pv_weight[j]
                continue

            # load maps
            pre_map = toio.load(os.path.join(pv_path, pv_maps[i - k - pv_no[0]]), device=device)['votes'].type(torch.float)
            nxt_map = toio.load(os.path.join(pv_path, pv_maps[i + k - pv_no[0]]), device=device)['votes'].type(torch.float)

            # ends towards current frame
            pre_fw_map, pre_fw_map_h = _flow_end_map(flow_data_fwd[depth - k:depth], pre_map, vector_mult=vector_mult, interpolate=interpolate, use_homography=key_homography_weight != 0)
            nxt_bk_map, nxt_bk_map_h = _flow_end_map(flow_data_bkw[depth - k:depth], nxt_map, vector_mult=vector_mult, interpolate=interpolate, use_homography=key_homography_weight != 0)

            # current frame towards ends
            cur_fw_map = _flow_cur_map(flow_data_fwd[depth:depth + k], nxt_map, vector_mult=vector_mult, interpolate=interpolate)
            cur_bk_map = _flow_cur_map(flow_data_bkw[depth:depth + k], pre_map, vector_mult=vector_mult, interpolate=interpolate)

            # gather votes
            vote_map += ((pre_fw_map + nxt_bk_map) * key_weight + (cur_fw_map + cur_bk_map) * cur_weight) * pv_weight[j]
            if key_homography_weight != 0:
                vote_map += (pre_fw_map_h + nxt_bk_map_h) * key_homography_weight * pv_weight[j]

        # construct new map
        voted = torch.argmax(vote_map.type(torch.float), 2)
        voted = classmap.logical(voted, noc)

        # save map
        np.savez_compressed(path, map=voted.cpu().numpy().astype(bool), votes=vote_map.cpu().numpy())

    # copy the rest
    if frame_filter is None:
        for i in list(range(depth)) + list(range(len(pv_maps) - depth, len(pv_maps))):
            copyfile(os.path.join(pv_path, pv_maps[i]), os.path.join(output_path, pv_maps[i]))

    # verbose
    # print('done!')

    return 0


def addext(pv_path, ext_path, output_path, ext_weight=1, frame_filter=None, device='cuda', overwrite=True):
    """
    Adds an external vote and reevaluates decisions.
    :param pv_path: String. Path to previous results folder.
    :param ext_path: String. Path to external input folder.
    :param output_path: String. Output folder.
    :param ext_weight: Float. External vote weight.
    :param frame_filter: Function. Receives frame number and returns 'True' for any frame that should be skipped. Useful for evaluation.
    :param device: py_torch device to run on.
    :param overwrite: Bool. Overwrite previous results.
    :return: 0 if completed successfully.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # meta
    pv_maps = sorted(os.listdir(pv_path))
    pv_no = [_frame_no(name) for name in pv_maps]
    shape = toio.load_array(os.path.join(pv_path, pv_maps[0]), device=device).shape
    basename = _basename(pv_maps[0])

    # process
    for i, p in enumerate(pv_no):
        noy, nox, noc = shape
        path = os.path.join(output_path, basename + '_{:06d}'.format(p) + '.npz')

        # only compute frames that aren't filtered
        if frame_filter is not None and frame_filter(p):
            continue

        # skip if present
        if not overwrite and os.path.exists(path):
            continue

        # verbose
        print('{:06d}'.format(p))

        # add votes
        vote_map = toio.load(os.path.join(pv_path, pv_maps[i]), device=device)['votes'].type(torch.float)
        ext_map = toio.load_array(os.path.join(ext_path, pv_maps[i]), device=device).type(torch.float)

        vote_map += ext_map * ext_weight
        vote_map /= (1 + ext_weight)

        # extract final map
        map = torch.argmax(vote_map, 2)
        map = classmap.logical(map, noc)

        # save
        np.savez_compressed(path, map=map.cpu().numpy().astype(bool), votes=vote_map.cpu().numpy())

    return 0


def denoise(flow_h5_path, pv_path, output_path, pv_series=None, surface_size=7, method='frames', interpolate=False,
            vector_mult=1, frame_filter=None, start_from=None, stop_at=None, device='cuda', overwrite=True):
    """
    _filters labels with a 3_d gaussian.
    :param flow_h5_path: String tuple. Tuple containing paths to (flow_forward.h5, flow_backward.h5)
    :param pv_path: String. Path to previous iteration folder.
    :param output_path: String. Output folder.
    :param pv_series: Int array. Frames to consider when constructing th 3_d volume to be filtered. Ex [0, 5] = 0 - current frame, 5 - frames at -5,+5 distance.
    :param surface_size: Float. Filter size on the YX axes, in pixels.
    :param method: String. Method for 3D volume construction. Possible values: 'self' - project current frame along flow vectors, 'frames' - project series frame.
    :param interpolate: Bool. Bilinearly interpolate coordinates during flow projection. Might be useful for low resolution data and long propagation distances.
    :param vector_mult: Float. Multiplies default flow vectors by the specified amount.
    :param frame_filter: Function. Receives frame number and returns 'True' for any frame that should be skipped. Useful for evaluation.
    :param start_from: Int. Start from frame number.
    :param stop_at: Int. Stop on frame number.
    :param device: py_torch device to run on.
    :param overwrite: Bool. Overwrite previous results.
    :return: 0 if completed successfully.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if pv_series is None:
        pv_series = [0]

    # meta
    pv_maps = sorted(os.listdir(pv_path))
    pv_no = [_frame_no(name) for name in pv_maps]
    noy, nox, noc = toio.load(os.path.join(pv_path, pv_maps[0]), device=device)['map'].shape
    basename = _basename(pv_maps[0])
    pv_series = sorted(pv_series)
    depth = pv_series[-1]
    nof = pv_no[-1]
    cur_times = sum([i == 0 for i in pv_series])
    pv_series = [i for i in pv_series if i > 0]
    depth_size = len(pv_series) * 2 + cur_times
    depth_sigma = depth_size / 2.5
    surface_sigma = surface_size / 2.5
    if start_from is None:
        start_from = pv_no[0]
    if stop_at is None or stop_at > pv_no[-1]:
        stop_at = pv_no[-1]

    # configure filter
    pad = util.pad((0, surface_size, surface_size))
    denoiser = blocks.GaussBlur3d((depth_size, surface_size, surface_size), (depth_sigma, surface_sigma, surface_sigma), pad=pad, device=device)

    # process
    for i in range(max(depth + pv_no[0], start_from), min(nof - depth + 1, stop_at)):
        path = os.path.join(output_path, basename + '_{:06d}'.format(i) + '.npz')

        # only compute frames that aren't filtered
        if frame_filter is not None and frame_filter(i):
            continue

        # overwrite
        if not overwrite and os.path.exists(path):
            continue

        # verbose
        print('{:06d}'.format(i))

        # load flows
        flow_data_fwd = h5py.File(flow_h5_path[0], 'r')
        flow_data_fwd = flow_data_fwd['flow'][i:i + depth]
        flow_data_fwd = torch.from_numpy(flow_data_fwd).to(device).flip(3)
        flow_data_bkw = h5py.File(flow_h5_path[1], 'r')
        flow_data_bkw = flow_data_bkw['flow'][i - depth:i]
        flow_data_bkw = torch.from_numpy(flow_data_bkw).to(device).flip(0).flip(3)

        # load map
        vote_map = toio.load(os.path.join(pv_path, pv_maps[i - pv_no[0]]), device=device)['votes'].type(torch.float)

        # 3d frame-time voxel votes
        if method == 'self':
            fw_maps = _flow_align(flow_data_fwd, vote_map, pv_series, vector_mult=vector_mult, interpolate=interpolate)
            bk_maps = _flow_align(flow_data_bkw, vote_map, pv_series, vector_mult=vector_mult, interpolate=interpolate)
        elif method == 'frames':
            fw_maps = torch.zeros((noy, nox, noc, len(pv_series)), device=device, dtype=torch.float)
            bk_maps = torch.zeros((noy, nox, noc, len(pv_series)), device=device, dtype=torch.float)
            for j, k in enumerate(pv_series):
                pre_map = toio.load(os.path.join(pv_path, pv_maps[i - k - pv_no[0]]), device=device)['votes']
                nxt_map = toio.load(os.path.join(pv_path, pv_maps[i + k - pv_no[0]]), device=device)['votes']
                bk_maps[..., j] = _flow_end_map(flow_data_fwd[depth - k:depth], pre_map, vector_mult=vector_mult, interpolate=interpolate)
                fw_maps[..., j] = _flow_end_map(flow_data_bkw[depth - k:depth], nxt_map, vector_mult=vector_mult, interpolate=interpolate)
            del pre_map, nxt_map
        elif method == 'noalign':
            fw_maps = torch.zeros((noy, nox, noc, len(pv_series)), device=device, dtype=torch.float)
            bk_maps = torch.zeros((noy, nox, noc, len(pv_series)), device=device, dtype=torch.float)
            for j, k in enumerate(pv_series):
                bk_maps[..., j] = toio.load(os.path.join(pv_path, pv_maps[i - k - pv_no[0]]), device=device)['votes']
                fw_maps[..., j] = toio.load(os.path.join(pv_path, pv_maps[i + k - pv_no[0]]), device=device)['votes']
        else:
            raise NotImplementedError
        votes = torch.cat((bk_maps.flip(3), vote_map.unsqueeze(-1).expand(-1, -1, -1, cur_times), fw_maps), dim=3)

        # denoise
        votes = votes.permute((2, 3, 0, 1))
        votes = votes.unsqueeze(0)
        votes = denoiser(votes).squeeze()
        votes = votes.permute((1, 2, 0))

        # extract final map
        map = torch.argmax(votes, 2)
        map = classmap.logical(map, noc)

        # save
        np.savez_compressed(path, map=map.cpu().numpy().astype(bool), votes=votes.cpu().numpy())

    # copy the rest
    if frame_filter is None:
        for i in list(range(depth)) + list(range(len(pv_maps) - depth, len(pv_maps))):
            copyfile(os.path.join(pv_path, pv_maps[i]), os.path.join(output_path, pv_maps[i]))

    # verbose
    # print('done!')

    return 0


def _flow_full(flow_data, vector_mult, interpolate=False):
    return flow.flow(flow_data, full_path=True, vector_mult=vector_mult, interpolate=interpolate)


def _end_map(flowed, class_map, use_homography=False):
    inverse = map2d.invert(flowed)
    flowed_map = map2d.apply(class_map, inverse)
    flowed_map[torch.isnan(flowed_map)] = 0
    flowed_map = flowed_map.type(torch.float)
    flowed_map_homography = None
    if use_homography:
        flowed_map_homography = _end_map_homography(flowed, class_map).type(torch.float).to(class_map.device)
    return flowed_map, flowed_map_homography


def _flow_end_map(flow_data, class_map, vector_mult=1, interpolate=False, use_homography=False):
    flowed = flow.flow(flow_data, vector_mult=vector_mult, interpolate=interpolate)
    return _end_map(flowed, class_map, use_homography=use_homography)


def _flow_cur_map(flow_data, class_map, vector_mult=1, interpolate=False):
    flowed = flow.flow(flow_data, vector_mult=vector_mult, interpolate=interpolate)
    flowed_map = map2d.apply(class_map, flowed).type(torch.float)
    return flowed_map


def _end_map_homography(flowed, class_map):
    noy, nox, noc = class_map.shape
    class_map = class_map.cpu().numpy()
    flowed = flowed.cpu().numpy()
    mapped = np.zeros((noy, nox, noc), dtype=np.float32)
    for c in range(noc):
        label, nob = ndimage.label(class_map[:, :, c], np.ones((3, 3)))
        for o in range(1, nob + 1):
            src = np.nonzero(label == o)
            if len(src[0]) < 4:
                continue
            dst = flowed[src[0], src[1]]
            src_cv = cv2.UMat(np.concatenate((src[0][:, None], src[1][:, None]), axis=1))
            dst_cv = cv2.UMat(dst)
            h, _ = cv2.findHomography(src_cv, dst_cv, method=cv2.LMEDS, ransacReprojThreshold=1.5)
            if h is not None:
                with warnings.catch_warnings():
                    warnings.filterwarnings('error')
                    try:
                        wrp = np.concatenate((src[0][:, None], src[1][:, None], np.ones((len(src[0]), 1))), axis=1) @ h.transpose()
                        wrp = wrp[:, 0:2] / wrp[:, 2:3]
                        wrp[:, 0] = np.maximum(np.minimum(np.round(wrp[:, 0]), noy - 1), 0)
                        wrp[:, 1] = np.maximum(np.minimum(np.round(wrp[:, 1]), nox - 1), 0)
                        wrp = wrp.astype(np.uint32)
                        mapped[wrp[:, 0], wrp[:, 1], c] = class_map[src[0], src[1], c]
                    except RuntimeWarning as e:
                        # print('homography c{0:d} o{1:d}:'.format(c, o) + str(e) + ', skipping!')
                        continue
    mapped = torch.from_numpy(mapped)
    return mapped


def _flow_align(flow_data, class_map, series, vector_mult=1, interpolate=False):
    flowed = flow.flow(flow_data, full_path=True, vector_mult=vector_mult, interpolate=interpolate)
    noy, nox, noc = class_map.shape
    nof = len(series)
    mapped = torch.zeros((noy, nox, noc, nof), device=class_map.device, dtype=class_map.dtype)
    for i in range(nof):
        mapped[..., i] = map2d.apply(class_map, flowed[..., series[i]])
    return mapped


def _frame_no(path):
    name = str.split(os.path.splitext(os.path.basename(path))[0], '_')
    no = int(name[-1])
    return no


def _basename(path):
    name = str.split(os.path.splitext(os.path.basename(path))[0], '_')
    basename = '_'.join(name[:-1])
    return basename
