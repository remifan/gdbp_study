import numpy as np
from collections import namedtuple
from tqdm.auto import tqdm
from commplax import comm
import labptptm2


Input = namedtuple('DataInput', ['y', 'x', 'w0', 'a'])


def load(src, lp, ch, rep, n_symbols=1500000):
    dat_grps, sup_grps = labptptm2.select(src, lp, ch, rep)

    inputs = []
    for dg, sg in tqdm(zip(dat_grps, sup_grps),
                       total=len(dat_grps),
                       desc='loading data',
                       leave=False):
        inputs.append(loader(dg, sg, n_symbols))
    return inputs


def loader(dat_grp, sup_grp, n_symbol=1500000):
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
    # 
    a = dict(zip(dat_grp.attrs.keys(), dat_grp.attrs.values())) # extract attributes
    a['CD'] = sup_grp.attrs['cd'] # measured CD
    sps = a['samplerate'] / a['baudrate'] # samples/symbol
    n_sample = np.round(n_symbol * sps).astype(int)

    # download data
    y = dat_grp['recv'][:n_sample]
    x = dat_grp['sent'][:n_symbol]
    w0 = sup_grp['nfo'][0] * sps # initial FO at symbol period used to initialize FOE

    # preprocessing
    y -= np.mean(y, axis=0) # block DC
    y = comm.normpower(y, real=True) / np.sqrt(2) # normalize inputs
    x /= comm.qamscale(a['modformat']) # normalize labels

    return Input(y, x, w0, a)

