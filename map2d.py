import torch
import util


def apply(source, mapping, interpolate=False):
    noy, nox, _ = mapping.shape

    iy = (mapping[:, :, 0] < 0) | (mapping[:, :, 0] > source.shape[0] - 1) | torch.isnan(mapping[:, :, 0])
    ix = (mapping[:, :, 1] < 0) | (mapping[:, :, 1] > source.shape[1] - 1) | torch.isnan(mapping[:, :, 1])
    ignore = iy | ix
    missing = ignore.any().item()  # TODO, slow

    # subpixel linear interpolation
    if interpolate:
        if missing:
            mapping[ignore] = 0
        mff = mapping.floor()
        mcf = torch.cat((mapping[:, :, 0:1].ceil(), mapping[:, :, 1:2].floor()), 2)
        mfc = torch.cat((mapping[:, :, 0:1].floor(), mapping[:, :, 1:2].ceil()), 2)
        mcc = mapping.ceil()
        blendy = mapping[:, :, 0] - mff[:, :, 0]
        blendx = mapping[:, :, 1] - mff[:, :, 1]
        sff = source[mff[:, :, 0].type(torch.long), mff[:, :, 1].type(torch.long)]
        scf = source[mcf[:, :, 0].type(torch.long), mcf[:, :, 1].type(torch.long)]
        sfc = source[mfc[:, :, 0].type(torch.long), mfc[:, :, 1].type(torch.long)]
        scc = source[mcc[:, :, 0].type(torch.long), mcc[:, :, 1].type(torch.long)]
        mapped = util.interp2d(sff, scf, sfc, scc, blendy, blendx)
        if missing:
            mapped = mapped.type(torch.float)
            mapped[ignore] = torch.tensor(float('nan'), device=source.device, dtype=torch.float)
    # rounded values
    else:
        if missing:
            mapping[ignore] = 0
        mapping = mapping.round().type(torch.long)
        mapped = source[mapping[:, :, 0], mapping[:, :, 1]]
        if missing:
            mapped = mapped.type(torch.float)
            mapped[ignore] = torch.tensor(float('nan'), device=source.device, dtype=torch.float)

    return mapped


def invert(mapping, dimensions=None):
    noy, nox, _ = mapping.shape
    if dimensions is None:
        ioy, iox = noy, nox
    else:
        ioy, iox = dimensions

    iy = (mapping[:, :, 0] < 0) | (mapping[:, :, 0] > ioy - 1) | torch.isnan(mapping[:, :, 0])
    ix = (mapping[:, :, 1] < 0) | (mapping[:, :, 1] > iox - 1) | torch.isnan(mapping[:, :, 1])
    ignore = iy | ix
    missing = ignore.any().item()

    y, x = torch.meshgrid([torch.arange(0, noy, device=mapping.device), torch.arange(0, nox, device=mapping.device)])
    idx = torch.cat([y[:, :, None], x[:, :, None]], 2).type(torch.float)

    inverted = util.nan((noy, nox, 2), device=mapping.device, dtype=torch.float)

    mapping = mapping.round().type(torch.long)
    if not missing:
        inverted[mapping[:, :, 0], mapping[:, :, 1], :] = idx
    else:
        inverted[mapping[~ignore][:, 0], mapping[~ignore][:, 1], :] = idx[~ignore]

    return inverted

