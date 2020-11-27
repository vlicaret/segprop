import torch
import util


# flow_data.shape = [no_steps, no_y, no_x, 2], with dim3 = [y, x] (flipped from FlowNet)
def flow(flow_data, locations=None, full_path=False, interpolate=False, track_edges=False, stride=1, vector_mult=1):
    nof, noy, nox, noc = flow_data.shape
    stops = nof // stride
    reshape = False

    # TODO error on non integer stops

    if locations is None:
        reshape = True
        y, x = torch.meshgrid([torch.arange(0, noy, device=flow_data.device), torch.arange(0, nox, device=flow_data.device)])
        locations = torch.cat([y.reshape(-1, 1), x.reshape(-1, 1)], 1)

    if full_path:
        flowed = torch.zeros((locations.shape[0], 2, stops + 1), device=locations.device)
    else:
        flowed = torch.zeros((locations.shape[0], 2, 2), device=locations.device)
    flowed[:, :, 0] = locations

    if track_edges:
        lost = torch.zeros(noy * nox, device=locations.device, dtype=torch.bool)
    f = 0
    for i in range(0, nof, stride):
        # current layer
        if not track_edges:
            loc = flowed[:, :, f]
        else:
            loc = flowed[~lost, :, f]

        # subpixel linear interpolation
        if interpolate:
            lff = loc.floor()
            lcf = torch.cat((loc[:, 0:1].ceil(), loc[:, 1:2].floor()), 1)
            lfc = torch.cat((loc[:, 0:1].floor(), loc[:, 1:2].ceil()), 1)
            lcc = loc.ceil()
            blendy = loc[:, 0] - lff[:, 0]
            blendx = loc[:, 1] - lff[:, 1]
            dff = flow_data[i, lff[:, 0].type(torch.long), lff[:, 1].type(torch.long), :] * vector_mult
            dcf = flow_data[i, lcf[:, 0].type(torch.long), lcf[:, 1].type(torch.long), :] * vector_mult
            dfc = flow_data[i, lfc[:, 0].type(torch.long), lfc[:, 1].type(torch.long), :] * vector_mult
            dcc = flow_data[i, lcc[:, 0].type(torch.long), lcc[:, 1].type(torch.long), :] * vector_mult
            dloc = util.interp2d(dff, dcf, dfc, dcc, blendy, blendx)
            if not track_edges:
                flowed[:, :, f + 1] = loc + dloc
            else:
                flowed[~lost, :, f + 1] = loc + dloc
        # rounded values
        else:
            dloc = flow_data[i, loc[:, 0].round().type(torch.long), loc[:, 1].round().type(torch.long), :] * vector_mult
            if not track_edges:
                flowed[:, :, f + 1] = loc + dloc
            else:
                flowed[~lost, :, f + 1] = loc + dloc

        # lost pixels
        if not track_edges:
            flowed[:, 0, f + 1] = util.bound(flowed[:, 0, f + 1], 0, noy - 1)
            flowed[:, 1, f + 1] = util.bound(flowed[:, 1, f + 1], 0, nox - 1)
        else:
            lost = (flowed[:, 0, f + 1] < 0) | (flowed[:, 0, f + 1] > noy - 1) |\
                   (flowed[:, 1, f + 1] < 0) | (flowed[:, 1, f + 1] > nox - 1) | lost
            flowed[lost, :, f + 1] = torch.tensor(float('nan'), device=flowed.device, dtype=flowed.dtype)

        if full_path:
            f = f + 1
        else:
            flowed[:, :, f] = flowed[:, :, f + 1]

    if not full_path:
        flowed = flowed[:, :, f]

    if reshape:
        flowed = flowed.reshape(noy, nox, 2, -1).squeeze()

    return flowed
