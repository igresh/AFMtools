from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from ForceCurveFuncs import resampleForceDataArray
import numpy as np

def plot_all_forcecurves(FCD, name='Untitled', save_path='./', alp=0.05):
    Extends, Retracts, ForceSepExtends, ForceSepRetracts = FCD

    fig, [[ax1, ax3], [ax2, ax4]] = plt.subplots(2,2, figsize=(8,5), sharex='col', sharey='col', tight_layout=True)

    fig.suptitle(name)
    for ax in [ax1, ax2, ax3, ax4]:
        ax.axhline(0, color='k', ls='--')

    for z, y in ForceSepExtends[0::1]:
        ax1.plot(z, y, color='xkcd:orange', alpha=alp)

    for z, y in ForceSepRetracts[0::1]:
        ax2.plot(z, y, color='xkcd:teal', alpha=alp)

    for z, y in Extends[0::1]:
        ax3.plot(z, y, color='xkcd:orange', alpha=alp)

    for z, y in Retracts[0::1]:
        ax4.plot(z, y, color='xkcd:teal', alpha=alp)


    ax2.axhline(0, color='k', ls='--')
    # ax2.set(xbound=(-15,150), ybound=(-1, 1), xlabel='separation, nm', ylabel='Force, nN')

    # ax4.set(xbound=(-15,150), ybound=(-1, 1), xlabel='displacement, nm', ylabel='Deflection, mV')

    fig.savefig(f'{save_path}/{name}.png')


    return fig, [[ax1, ax3], [ax2, ax4]]


def make_QC_video(FCD, subdf, max_pulloffs_plotted=7, upto=None, save_name='Untitled.mkv', save_path='./'):
    """
    Creates a animation showing each forcecurve (extend and retract) with single-molecule fits overlaid.
    FCD   : the force curve dataset, where index 2 is force vs. sep extends and index 3 is force vs. sep retracts

    subdf : the dataframe containing all the pull off events associated with FCD

    """
    ForceSepExtends = resampleForceDataArray(FCD[2])
    ForceSepRetracts = resampleForceDataArray(FCD[3])


    def init():
        xmax = np.nanmax(ForceSepExtends[:,0])
        xmin = np.nanmin(ForceSepExtends[:,0])
        ymax = np.nanmax(ForceSepRetracts[:,1])
        ymin = np.nanmin(ForceSepRetracts[:,1])
        ax1.set_xlim(xmin, xmax)
        ax1.set_ylim(ymin, 0.1)

        ax1.set(ylabel='Force, nN')
        ax2.set(ylabel='Force, nN', xlabel='Tip-substrate separation, nm')


    def update(frame, dataframe):
        gb = dataframe.groupby('index')

        idx, [[x1, y1], [x2, y2]] = frame
        text.set_text(idx)
        ln1.set_data(x1, y1)
        ln2.set_data(x2, y2)

        for mod_artist, data_artist, text_artist in zip(WLC_model_artists, WLC_data_artists, WLC_text_artists):
            mod_artist.set_data([], [])
            data_artist.set_data([], [])
            text_artist.set_text('')

        if idx in gb.groups:
            for mod_artist, data_artist, text_artist, df_idx in zip(WLC_model_artists, WLC_data_artists, WLC_text_artists, gb.groups[idx]):
                df_row = subdf.loc[df_idx]
                mod_artist.set_data(df_row['pull-off model'][0], -df_row['pull-off model'][1])
                data_artist.set_data(df_row['pull-off data'][0], -df_row['pull-off data'][1])

                text_artist.set_x(df_row['pull-off data'][0][-1])
                text_artist.set_y(np.min(-df_row['pull-off data'][1])-0.2)
                text_artist.set_text(np.round(df_row['Contour Length'], 1))


    # Prepare figure
    fig, [ax1, ax2] = plt.subplots(2,1, figsize=(4,6), sharex=True, sharey=True, tight_layout=True)
    xdata, ydata = [], []
    ln1, = ax1.plot([], [], 'k.')
    ln2, = ax2.plot([], [], 'k.')

    WLC_model_artists = []
    WLC_data_artists = []
    WLC_text_artists = []

    for i in range(max_pulloffs_plotted):
        WLC_model_artists.append(ax2.plot([], [], color='b', zorder=10)[0])
        WLC_data_artists.append(ax2.plot([], [], 'r.')[0])
        WLC_text_artists.append(ax2.text(0, 0, s='', ha='center', va='top'))

    text = ax1.text(0.98, 0.02, s=0, ha='right', va='bottom', transform=ax1.transAxes)

    if upto is None:
        upto = ForceSepExtends.shape[0]

    ani = FuncAnimation(fig, update, frames=enumerate(zip(ForceSepExtends[:upto], ForceSepRetracts[:upto])),
                        init_func=init, blit=False, repeat=False, save_count=upto, fargs=[subdf],)

    ani.save(f'{save_path}/{save_name}', dpi=300, writer='ffmpeg')
