import numpy as np
import glob
import os
from scipy import stats
from scipy.signal import savgol_filter
import copy
import warnings
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def splitExtendRetract(ForceData):
    """
    ForceData : a 2xN array, where the [0] is the Z-piezo position and [1] is the deflection voltage
    """
    
    maxIndex  = np.argmax(ForceData[1])
    Extend    = ForceData[:,0:maxIndex]
    Retract   = np.flip(ForceData[:,maxIndex:], axis=1)
    
    return Extend, Retract


def extractHardContactRegion(ForceData, SplitFirst=False):
    """
    Extracts compliance regions from force data. If the data contains both extension and retraction,
    split first should be true. 
    
    ForceData : a 2xN array, where the [0] is the Z-piezo position and [1] is the deflection voltage

    returns:
    data from the compliance region. If ForceData included both an extension and retraction curve, then
    will return compliance data from both curves.
    
    """
    
    if SplitFirst:
        Extend, Retract = splitExtendRetract(ForceData)
        return extractCompli fanceRegion(Extend, False), extractComplianceRegion(Retract, False)
    
    else:
        MaxIndex = np.argmax(ForceData[1][ForceData[1]>0])
        MinIndex = np.argmin(ForceData[1][ForceData[1]>0])
        
        if MaxIndex > MinIndex: # This shouldn't be required if the sanitisation in splitExtendRetract works properly
            return ForceData[:,MinIndex:]
        else:
            return ForceData[:,:MinIndex]


def parseDirectory(Directory):
    """
    Opens all the files in a directory, returns an iterable such that you can use something like
    
    for Xfilename, Yfilename in openDirectory('CleanSilica2/'):

    """

    Xfilenames = glob.glob(os.path.join(Directory, '*ZSnsr.txt'))
    # Yfilenames = glob.glob(os.path.join(Directory, '*DeflV.txt'))
    Yfilenames = []
    for xfn in Xfilenames:
        Yfilenames.append(xfn.replace('ZSnsr', 'DeflV'))
    # assert len(Xfilenames) == len(Yfilenames)

    return zip(Xfilenames, Yfilenames)


def RemoveBaseline(ForceData, approachFraction=0.5):
    X, Y = ForceData
    Xrange = np.max(X) - np.min(X)
    Partition_mask = X > np.min(X) + Xrange*approachFraction
    

    smoothygrad = np.abs(savgol_filter(Y, int(len(Y)/50), 1, deriv=1))
    gradient_cutoff = 0.25*np.average(smoothygrad)
    gradient_mask = smoothygrad < gradient_cutoff
    
    mask = np.logical_and(Partition_mask, gradient_mask)
        
    m, c, r_value, p_value, std_err = stats.linregress(X[mask],Y[mask])
    
    baseline = X*m+c
    OffsetY = Y - baseline
    
    if np.abs(m) > 100000:
        warnings.warn("Baseline gradient exceeds 100000 - check approachFraction")
        return None        
    else:
        return copy.deepcopy(np.array([X, OffsetY]))


def zeroForceCurves(ForceData, forceThreshold=3e-8):
    """
    Performs a zeroing on force vs. separation curves
    
    (NOT displacement / z-piezeo position curves - use "ConvertToForceVSep" to convert
    to force vs. separation first)
    """
    if ForceData.ndim == 3:
        for FD in ForceData:
            mask = FD[1] > forceThreshold
            FD[0] -= np.mean(FD[0][mask])

    else:
        mask = ForceData[1] > forceThreshold
        ForceData[0] -= np.mean(ForceData[0][mask])

    return ForceData
    
    


def findComplianceEdge(compliance, threshold=3, complianceMinDeflV=0.4, returnVal=True):
    """
    Works backwards from maximum deflection to find the 'edge' of the compliance region, presumably this is the
    Z position of the substrate (or close to it).
    
    Threshold: Number of standard deviations that the signal needs to deviate from the mean compliance region gradient 
    to be considered outside of the compliance region.
    
    complianceMinDeflV: Deflection voltage above which we are assumed to be in the compliance region.
    
    returnVal: If True returns the z value where he compliance region starts. If false returns the index.
    
    """
    x, y  = compliance
    ydiff = savgol_filter(y,15,1, deriv=1)
    compliance_gradient = np.mean(ydiff[y>complianceMinDeflV])
    compliance_std = np.std(ydiff[y>complianceMinDeflV])
    
    complianceEdgeMask = ydiff[y<complianceMinDeflV]-compliance_gradient<-compliance_std*threshold

    if np.any(complianceEdgeMask):
        complianceEdgeIdx = np.argwhere(complianceEdgeMask)[-1]
    else:
        complianceEdgeIdx = np.array([0])

    
    if returnVal:
        return compliance[:, complianceEdgeIdx]
    else:
        return complianceEdgeIdx[0]


    
def calculateSensitivity(ForceData, complianceMinDeflV=0.2):
    compliance = extractHardContactRegion(ForceData)

    mask = compliance[1]>complianceMinDeflV
    ExtendSen = -1/stats.linregress(compliance[0, mask], compliance[1, mask]).slope

    return ExtendSen

def ConvertToForceVSep(ForceData, sensitivity=0, spring_constant=1):
    """
    Converts the data to separation (if spring constant != 1, also converts to force)
    """       
    
    if sensitivity==None:
        sensitivity = calculateSensitivity(ForceData)

    ForceData = copy.deepcopy(ForceData)

    if ForceData.ndim == 3:
        ForceData[:,1] *= sensitivity
        ForceData[:,0] += ForceData[:, 1]
        ForceData[:,1] *= spring_constant
    else:
        ForceData[1] *= sensitivity
        ForceData[0] += ForceData[1]
        ForceData[1] *= spring_constant

    return ForceData
  
    
def find_SMpulloffs(ForceSepRetract, verbose=False):
    """
    Returns values in distance from susbtrate and adhesion force (nN) (as a positive quantity)
    
    """
    x, y = ForceSepRetract
    x = x[~np.isnan(y)]
    y = y[~np.isnan(y)]

    # Smooth the y-signal to remove outliers when masking
    newy = savgol_filter(y, int(len(x)/80), 1)
    mask = np.logical_and(x>5, newy < -0.05)

    PO_x = x[mask]
    PO_y = y[mask]
    PO_newy = newy[mask]
    
    xdiff = np.abs(np.diff(PO_x))
    ydiff = np.diff(PO_newy)
    splits = np.argwhere(np.logical_or(xdiff>4, ydiff>0.001))[:,0]
    splits = np.concatenate([np.array([0]), splits, np.array([len(xdiff)-10])])
    
    split_SM_curves = []
    discarded_SM_curves = []

    for [sx1, sx2] in zip(splits[:-1], splits[1:]):
        start_slice = sx1+1
        
        end_slice = sx2-1
        if end_slice > len(PO_x):
            end_slice = len(PO_x)-1
    
        tempx = PO_x[start_slice:end_slice].copy()
        tempy = PO_y[start_slice:end_slice].copy()
        
        if len(tempx) > 30: # If the array has less than 30 entries, then don't worry about it (keep in mind we're padding)
            
            # We only take values with a negative slope (overall), because we just want the
            # pull-off events.
            
            a, b = np.polyfit(tempx, tempy, 1)
            
            run = tempx[-1] - tempx[0]
            if a*run < -0.015:
                split_SM_curves.append([abs(tempx), abs(tempy)])
            else:
                discarded_SM_curves.append([abs(tempx), abs(tempy)])

                
    if verbose:
        return split_SM_curves, discarded_SM_curves, [[PO_x, PO_y, PO_newy]]
    else:
        return split_SM_curves
    
    
    
def make_QC_video(FCD, subdf, max_pulloffs_plotted=7, upto=None, save_name='Untitled.mkv', save_path='./'):
    """
    FCD   : the force curve dataset, where index 2 is force vs. sep extends and index 3 is force vs. sep retracts

    subdf : the dataframe containing all the pull off events associated with FCD
    
    """
    ForceSepExtends = FCD[2]
    ForceSepRetracts = FCD[3]
    

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