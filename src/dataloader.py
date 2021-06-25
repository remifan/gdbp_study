import numpy as np
from collections import namedtuple
from commplax import comm
import labptptm2

labptptm2.config(supdata=True, dump_dir='./dataset')

Dataset = namedtuple('Data', ['y', 'x', 'w0', 'a'])


def loaddata(src: int, lp: int, ch: int, rep: int, num=1500000):
    ''' dataloader of LabPtPTm2 dataset
        more info: https://github.com/remifan/LabPtPTm2

        Args:
            src: QRBS source index
            lp: launched power in dBm
            ch: channel index
            rep: sample index
            num: number of symbols to load

        Returns:
            dataset stored in namedtuple: (2 sample/symbol waveforms,
                                           aligned sent symbols,
                                           monitored initial FO,
                                           data attributes)
    '''

    with labptptm2.file(src, lp, ch, rep) as fd:
        y = fd['recv'][:num * 2]
        x = fd['sent'][:num]
        a = dict(zip(fd.attrs.keys(), fd.attrs.values())) # extract hdf attributes
    y -= np.mean(y, axis=0) # block DC
    y = comm.normpower(y, real=True) / np.sqrt(2) # normalize power
    x /= comm.qamscale(a['modformat']) # rescale ground truth

    with labptptm2.file(src, lp, ch, rep, supdata=True) as fd:
        nfo = fd['nfo'][...] # coarsely monitored frequency offset evolution normalized to sample period
        a['CD'] = fd.attrs['cd'] # measured CD

    w0 = nfo[0] * 2 # initial FO at symbol period used to initialize FOE

    return Dataset(y, x, w0, a)

# import os
# import h5py
# from pathlib import Path
# import numpy as np
# from commplax import comm
# import numpy as np
# from collections import namedtuple


# Dataset = namedtuple('Data', ['y', 'x', 'w0', 'a'])

# datadir = './dataset'


# def loaddata(src: int, lp: int, ch: int, idx: int, num=1500000):
#     ''' data loader

#         Args:
#             src: QRBS source number
#             lp: launched power in dBm
#             ch: channel index
#             idx: sample index

#         Returns:
#             dataset stored in tuple: (2 sample/symbol waveforms, aligned sent symbols,
#             monitored FO evolution, data attributes)
#     '''

#     dpath = _datapath(src, lp, ch, idx)

#     with h5py.File(dpath, 'r') as dhf:
#         y = dhf['recv'][:num * 2]
#         x = dhf['sent'][:num]

#         a = dict(zip(dhf.attrs.keys(), dhf.attrs.values())) # extract hdf attributes
#         # preprocess: block DC, normalization
#         y -= np.mean(y, axis=0)
#         y = comm.normpower(y, real=True) / np.sqrt(2)
#         x /= comm.qamscale(a['modformat'])

#         # load supplementary (monitored FO and CD)
#         spath = _supdatapath(src, lp, ch, idx)
#         with h5py.File(spath, 'r') as sdhf:
#             nfo = sdhf['nfo'][...]
#             a['CD'] = sdhf.attrs['cd']

#         # convert monitored FO to FOE multiplier
#         # f = np.exp(-1j * np.cumsum(nfo)).astype(np.complex64)
#         w0 = nfo[0] * 2 # normalized by symbol period
#     return Dataset(y, x, w0, a)


# def _datapath(src, lp, ch, idx):
#     sample = 'data/1125km_src%d/%ddBm_ch%d_%d.h5' % (src, lp, ch, idx)
#     return os.path.join(datadir, sample)


# def _supdatapath(src, lp, ch, idx):
#     supdatadir = os.path.join(datadir, 'supplementary_data/1125km_src%d' % src)

#     Path(supdatadir).mkdir(parents=True, exist_ok=True)
#     sample = '%ddBm_ch%d_%d.h5' % (lp, ch, idx)
#     return os.path.join(supdatadir, sample)

