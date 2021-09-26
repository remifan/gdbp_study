import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from commplax import plot as cplt, comm


def lp_vs_q(lp, q, ax=None, **kwargs):
    if ax is None:
        plt.figure()
        ax = plt.gca()
    ax.plot(lp, q, '-o', **kwargs)
    ax.legend()
    ax.set_xlabel('Launched Power (dBm)')
    ax.set_ylabel('Q-facotr (dB)')
    # px.line(q_ch1, width=800, height=600, markers=True, template='plotly_white')


def wireframe_cmap(ax, X, Y, Z, cmap=plt.cm.viridis, offset=[0, 0], label=""):
    norm = plt.Normalize(Z.min() + offset[0], Z.max() + offset[1])
    colors = cmap(norm(Z))
    surf = ax.plot_surface(X,
                           Y,
                           Z,
                           facecolors=colors,
                           shade=False,
                           label=label)
    try:
      # matplotlib >= 3.3.3
      surf._facecolors2d = surf._facecolor3d
      surf._edgecolors2d = surf._edgecolor3d
    except AttributeError:
      surf._facecolors2d = surf._facecolors3d
      surf._edgecolors2d = surf._edgecolors3d
    surf.set_facecolor((0, 0, 0, 0))
    surf.set_facecolor((0, 0, 0, 0))


def loss(loss, ax=None, label=None, alpha=0.4):
    if ax is None:
        plt.figure()
        ax = plt.gca()
    loss_mean = np.convolve(loss, np.ones(100) / 100, mode='same')
    p = ax.plot(loss[:], alpha=alpha, label=label)
    ax.plot(loss_mean[:-50], color=p[0].get_color())
    ax.set_xlabel('iteration')
    ax.set_ylabel('MSE')
    ax.legend()


def gdbp_params(params, vertical=False, sr=72, bw=60, dpi=200):
    if vertical:
        fig, axs = plt.subplots(3, 1, figsize=(4, 8), dpi=dpi)
    else:
        fig, axs = plt.subplots(1, 3, figsize=(10, 2.5), dpi=dpi)
    param_D = np.stack(
        [params[-5]['fdbp_0']['DConv_%d' % i]['kernel'] for i in range(3)])
    param_N = np.stack(
        [params[-5]['fdbp_0']['NConv_%d' % i]['kernel'] for i in range(3)])
    param_R = params[-5]['RConv']['kernel']

    ax = axs[0]
    cplt.desc_filter(*comm.firfreqz(param_D[:, :, 0], sr=sr, bw=bw),
                     ax=ax,
                     legend=['Step 1', 'Step 2', 'Step 3'],
                     colors=plt.cm.RdPu_r(np.linspace(0, 0.8, len(param_D))))
    ax.set_xlabel('freq. (GHz)')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    ax = axs[1]
    cplt.desc_filter(*comm.firfreqz(param_N[:, :, 0, 0], sr=sr, bw=bw),
                     ax=ax,
                     legend=['Step 1', 'Step 2', 'Step 3'],
                     colors=plt.cm.RdPu_r(np.linspace(0, 0.8, len(param_D))),
                     Hermitian=True)
    ax.set_xlabel('freq. (GHz)')
    ax.set_ylim([-0.05, 0.25])
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    ax = axs[2]
    cplt.desc_filter(*comm.firfreqz(param_R[:, 0],
                                    sr=sr,
                                    bw=bw),
                     ax=ax)
    ax.set_xlabel('freq. (GHz)')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    fig.tight_layout()
    return fig


