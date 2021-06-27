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

