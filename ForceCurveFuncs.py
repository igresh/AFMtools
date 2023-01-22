import numpy as np
import glob
import os
from scipy import stats, optimize
from scipy.signal import savgol_filter
import copy
import warnings
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from itertools import compress
from numpy.polynomial import polynomial


def parabola(x, a, b, c, LR):
    return a*(x-LR)**2 + b*(x-LR) + c

def consecutive(data, stepsize=1):
    "Return groups of consecutive elements"
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)


def process_zpos_vs_defl(zpos, defl, metadict=None, defl_units='nm', zpos_units='nm', zpos_negative=True, max_to_process=None,
                         number_of_curves_before_equil=1, override_involS=False, override_spring_constant=False, debug=False):
    """
    Processes raw Z piezo position and corresponding deflection data, returning normalised Z position and deflection curves, as well as
    tip-separation and force curves.
    
    
    Parameters
    ----------
    zpos (list):
        list of Z piezo position, provided by the AFM. Must be the same length as defl. Each entry within zpos must
        also be the same length as the corresponding entry in defl.

    defl (list):
        list of deflection, in either nanometers or milli volts. Ensure that defl_units are set correctly. Must be the
        same length as zpos. Each entry within defl must also be the same length as the corresponding entry in zpos.
        
    metadict (dict):
        dictionary containing metadata relating to the force curve map. At a minimum, must contain:
            'SpringConstant' (float) : spring constant of the cantilever (will default to 1 if not supplied)
            'InvOLS' (float) : optical sensitivity of the system (required only if defl_units='nm')
                 
    zpos_units (string):
        Units of defl input. Must be one of 'm' (meters), 'mm' (millimeters), 'um' (micrometers), or 'nm' (nanometers).
        Output will always be in nanometers
    
    zpos_negative (bool):
        Whether or not the zpos is given as a negative value (distance from the substrate) or a positive value (distance to
        the substrate) 

    defl_units (string):
        Units of defl input. Must be one of 'm' (meters), 'nm' (nanometers), 'V' (volts), or 'mV' (milli volts).
        Output defl will always be in nanometers
    
    max_to_process (int, or None):
        Maximum number of zpos and defl pairs to process. Useful for testing.
        
    number_of_curves_before_equil (int):
        Number of forcecurves to discard at the start of zpos and defl, useful as the first force curves can sometimes 
        be erroneous. Takes zpos = zpos[number_of_curves_before_equil:] and defl = defl[number_of_curves_before_equil:]
        
    override_involS (False, or float):
        Whether or not to use a fixed optical sensitivity or to calculate the optical sensitivity from the constant
        compliance region of each force curve. If False, the sensitivity is calculated from each force curve. If not False,
        then the value of override_involS is used as the involS.
    
    override_spring_constant (False, or float):
        Whether or not to override the spring constant in metadict. If False, does not override. If not False, the value of
        override_spring_constant is used as the spring constant.

    debug (bool):
        *Not implimented* If True, will display additional output usefull for debugging.
    
    """
    
    # initialise variables
    ExtendsXY = []
    RetractsXY = []
    ExtendsForce = []
    RetractsForce = []
    Esens = []
    Rsens = []

    discard_count = 0
    
    
    # Determine global parameters
    if override_involS:
        invOLS = override_involS
    else:
        if metadict == None:
            invOLS = 1
        elif 'InvOLS' in metadict.keys():
            invOLS = float(metadict['InvOLS'])
        else:
            invOLS = 1

    if override_spring_constant:
        spring_constant = override_spring_constant
    else:
        if metadict == None:
            spring_constant = 1
        elif 'SpringConstant' in metadict.keys():
            spring_constant = float(metadict['SpringConstant'])
        else:
            spring_constant = 1
            
    print (f"InvOLS: {invOLS}, Spring constant: {spring_constant}")
    

    # trim data
    if max_to_process == None:
        max_to_process = len(zpos)
    else:
        max_to_process += number_of_curves_before_equil

    zpos = zpos[number_of_curves_before_equil:max_to_process]
    defl = defl[number_of_curves_before_equil:max_to_process]

    
    # loop through zpos and defl
    for idx, [z, d] in enumerate(zip(zpos, defl)):
        if not is_data_sanitary([z, d]):
            print (f"entry {number_of_curves_before_equil + idx} dead on arrival")
            discard_count += 1
            continue
    
    
        # Convert units
        d = convert_defl_to_mV(d, defl_units, invOLS)
        z = convert_zpos_to_nm(z, zpos_units, flip=zpos_negative)
        XYdata = np.array([z,d])

    
        # Remove dwell drift
        XYdata = remove_dwell_drift(XYdata)
        if not is_data_sanitary(XYdata):
            print (f"entry {number_of_curves_before_equil + idx} Failed on dwell drift removal")
            discard_count += 1
            continue
        

        # Split data into approach and retract
        ExtendXY, RetractXY = split_and_normalize(XYdata)
        if not is_data_sanitary([ExtendXY, RetractXY]):
            print (f"entry {number_of_curves_before_equil + idx} Failed on split and normalize")
            discard_count += 1
            continue


        # Calculate sensitivity
        Esen = calculateSensitivity(ExtendXY)
        Rsen = calculateSensitivity(RetractXY)
        if np.isnan(Esen) or np.isnan(Rsen):
            print (f"entry {number_of_curves_before_equil + idx} Failed to find constant compliance sensitivity")
            discard_count += 1
            continue

            
        # Convert to force vs. separation
        average_sens = (Esen + Rsen)/2
        ExtendForce  = ConvertToForceVSep(ExtendXY, sensitivity=average_sens, spring_constant=float(metadict['SpringConstant']))
        RetractForce = ConvertToForceVSep(RetractXY, sensitivity=average_sens, spring_constant=float(metadict['SpringConstant']))
        if not is_data_sanitary([ExtendForce, RetractForce]):
            print (f"entry {number_of_curves_before_equil + idx} Failed on Force conversion")
            discard_count += 1
            continue


        # Correct remaining baseline curvature
        ExtendForce, RetractForce = clean_forceData(ExtendForce, RetractForce, force_std_thresh=1)
        ExtendForce, RetractForce = RemoveBaseline_nOrder(ExtendForce, approachFraction=0.05, bonus_ForceData=RetractForce)
        if not is_data_sanitary([ExtendForce, RetractForce]):
            print (f"entry {number_of_curves_before_equil + idx} Failed on baseline curvature correction")
            discard_count += 1
            continue


        # Clean up data, one last time
        ExtendForce, RetractForce = clean_forceData(ExtendForce, RetractForce, force_std_thresh=0.02)
        if not is_data_sanitary([ExtendForce, RetractForce]):
            print (f"entry {number_of_curves_before_equil + idx} Failed on cleanup")
            discard_count += 1
            continue


        # Append processed data to lists
        ExtendsXY.append(ExtendXY)
        RetractsXY.append(RetractXY)
        ExtendsForce.append(ExtendForce)
        RetractsForce.append(RetractForce)
        Esens.append(Esen)
        Rsens.append(Rsen)
        
        if debug:
            userinput = input("Do you want to continue?")
            if userinput.lower() != 'y':
                break
        
    # Get rid of data that deviates from the mean sensitivity
    AvExSens = np.mean(Esens)
    StdExSens = np.std(Esens)
    AvRetSens = np.mean(Rsens)
    StdRetSens = np.std(Esens)

    ExSensMask = np.logical_and(Esens > AvExSens - 2*StdExSens, Esens < AvExSens + 2*StdExSens)
    RetSensMask = np.logical_and(Rsens > AvRetSens - 2*StdRetSens, Rsens < AvRetSens + 2*StdRetSens)
    SensMask = np.logical_and(ExSensMask, RetSensMask)
    
    number_excluded_by_sens = np.sum(np.logical_not(SensMask))
    if number_excluded_by_sens != 0:

        print (f"The following were excluded on the basis of their optical sensitivity being more than two standard deviations away from the mean:\n {number_of_curves_before_equil +  np.ravel(np.argwhere(np.logical_not(SensMask)))}")

        discard_count += number_excluded_by_sens

        ExtendsForce  = list(compress(ExtendsForce, SensMask))
        RetractsForce = list(compress(RetractsForce, SensMask))
        ExtendsXY     = list(compress(ExtendsXY, SensMask))
        RetractsXY    = list(compress(RetractsXY, SensMask))

        # Recalculate the mean sensitivity
        AvExSens = np.mean(np.array(Esens)[SensMask])
        AvRetSens = np.mean(np.array(Rsens)[SensMask])


    # Print stuff that you might want to know
    print("Extend Sensitivity: " + str(AvExSens) + " nm/V")
    print("Retract Sensitivity: " + str(AvRetSens) + " nm/V")
    print (f'{discard_count}/{len(zpos)-1} discarded' )

    return [ExtendsXY, RetractsXY, ExtendsForce, RetractsForce]

    
def convert_defl_to_mV(defl, defl_units, invOLS=None):
    if defl_units == 'nm':
        assert invOLS != None , "if defl_units is a length, involS must be supplied"
        defl = defl/invOLS
    elif defl_units == 'm':
        assert invOLS != None , "if defl_units is a length, involS must be supplied"
        defl = defl/invOLS
    elif defl_units == 'mV':
        defl = defl
    elif defl_units == 'V':
        defl = defl/1000
    else:
        raise Exception("defl_units not recognised")
        
    return defl


def convert_zpos_to_nm(zpos, zpos_units, flip=True):
    if flip:
        mult = -1
    else:
        mult = 1

    if zpos_units == 'nm':
        zpos *= mult
    elif zpos_units == 'um':
        zpos *= mult*1e3 
    elif zpos_units == 'mm':
        zpos *= mult*1e6 
    elif zpos_units == 'm':
        zpos *= mult*1e9 
    else:
        raise Exception("zpos_units not recognised")
        
    return zpos


def is_data_sanitary(data):
    """
    Takes Force vs Separation data and determines if theres anything seriously wrong with it.
    """
    
    for datum in data:
        if np.any(datum) == None:
            return False
        elif np.any(np.isnan(datum)):
            return False
        elif np.ndim(datum) == 1 and datum.shape[0] < 100:
            return False
        elif np.ndim(datum) == 2 and datum.shape[1] < 100:
            return False

    return True


def clean_forceData(ApproachForceData, RetractForceData, force_std_thresh=0.01):
    mask = (np.abs(ApproachForceData[1])<0.15)
    newApproachForceData = ApproachForceData.T[mask].T
    newApproachForceData = zeroForceCurves(newApproachForceData)

    mask = (np.abs(RetractForceData[1])<0.15)
    newRetractForceData = RetractForceData.T[mask].T
    newRetractForceData = zeroForceCurves(newRetractForceData)

    mask = newApproachForceData[0] > 5
    std_non_compliance = np.std(newApproachForceData[1][mask])
    
    if std_non_compliance > force_std_thresh: # large fluctuations in approach curves (which should not be there):
        return None, None
    else:
        return newApproachForceData, newRetractForceData
    


def split_and_normalize(XYdata):
    ExtendXY, RetractXY = splitExtendRetract(XYdata)
    
    if ExtendXY.shape[1] < 100 or RetractXY.shape[1] < 100:
        return None, None
    
    ExtendXY  = RemoveBaseline_Linear(ExtendXY, approachFraction=0.3)
    RetractXY = RemoveBaseline_Linear(RetractXY, approachFraction=0.3)

    if ExtendXY is None or RetractXY is None:
        return None, None

    else:
        try:
            ExtendXY = np.flip(ExtendXY, axis=1)
            RetractXY = np.flip(RetractXY, axis=1)

            ExtendXY = ExtendXY[:,20:-20]
            RetractXY = RetractXY[:,20:-20]
            
            # Have hard contact occur at 0 nm
            threshold = 0.1
            idx = np.where(ExtendXY[1]>threshold)[0][0]
            ExtendXY[0]  -= ExtendXY[0][idx]

            idx = np.where(RetractXY[1]>threshold)[0][0]
            RetractXY[0]  -= RetractXY[0][idx]

            return ExtendXY, RetractXY
        except:
            return None, None
    


def splitExtendRetract(ForceData):
    """
    ForceData : a 2xN array, where the [0] is the Z-piezo position and [1] is the deflection voltage
    """
    
    maxIndex  = np.argmax(ForceData[1])
    Extend    = ForceData[:,0:maxIndex]
    Retract   = np.flip(ForceData[:,maxIndex:], axis=1)
    
    return Extend, Retract



def extractHardContactRegion(ForceData, SplitFirst=False, quantile=0.90):
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
        return extractHardContactRegion(Extend, False), extractHardContactRegion(Retract, False)
    
    else:
        q = np.nanquantile(ForceData[1], 0.9, axis=0)
        if q < 0:
            q=0

        IndexAboveThreshold = np.argwhere(ForceData[1]>q)
        FirstConsecutiveSet = consecutive(IndexAboveThreshold.flatten())[0]
        MaxIndex = FirstConsecutiveSet[0]
        MinIndex = FirstConsecutiveSet[-1]
        
        if MaxIndex > MinIndex: # This shouldn't be required if the sanitisation in splitExtendRetract works properly
            return ForceData[:,MinIndex:]
        else:
            return ForceData[:,:MinIndex]



def remove_dwell_drift(XYdata):
    """
    Sometimes there will be a deflection drift without a change in raw Z position. This will confuse the code later on.
    We remove it by assuming that the derivative of Z-sensor position in this area will be close to zero. 
    
    Split extend/retract speeds may cause problems, because the Z sensor velocity threshold is taken from the average
    of the Z sensor velocity over the course of the experiment. As long as the retract is not 20 times slower than the
    approach, this code should be fine.
    """
    raw = XYdata[0]
    
    window_length =  int(raw.shape[0]/100)
    if window_length%2 == 0:
        window_length += 1

    newraw = savgol_filter(raw,window_length, 1) 
    newrawgrad = np.gradient(newraw)

    mask = np.abs(newrawgrad)>np.quantile(np.abs(newrawgrad), 0.5)/2
    
    return XYdata.T[mask].T



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



def RemoveBaseline_Linear(ForceData, approachFraction=0.5):
    if np.any(ForceData) == None:
        return None

    X, Y = ForceData
    Xrange = np.max(X) - np.min(X)
    Partition_mask = X > np.min(X) + Xrange*approachFraction
    
    window_length = int(len(Y)/50)
    if window_length%2 == 0:
        window_length += 1

    smoothygrad = np.abs(savgol_filter(Y, window_length, 1, deriv=1))
    gradient_cutoff = 0.25*np.average(smoothygrad)
    gradient_mask = smoothygrad < gradient_cutoff
    
    mask = np.logical_and(Partition_mask, gradient_mask)
        
    if np.sum(mask) > 100: #Need at least 100 eligible datapoints to continue
        m, c, r_value, p_value, std_err = stats.linregress(X[mask],Y[mask])

        baseline = X*m+c
        OffsetY = Y - baseline

        if np.abs(m) > 100000:
            warnings.warn("Baseline gradient exceeds 100000 - check approachFraction")
            return None        
        else:
            return copy.deepcopy(np.array([X, OffsetY]))
    else:
        return None


    
def RemoveBaseline_nOrder(ForceData, order=3, approachFraction=0.2, bonus_ForceData=None):
    if np.any(ForceData) == None:
        return None, None

    X, Y = ForceData
    Xrange = np.max(X) - np.min(X)
    Partition_mask = X > np.min(X) + Xrange*approachFraction
    
    window_length = int(len(Y)/50)
    if window_length%2 == 0:
        window_length += 1

    smoothygrad = np.abs(savgol_filter(Y, window_length, 1, deriv=1))
    gradient_cutoff = 0.5*np.median(smoothygrad[Partition_mask])
    
    gradient_mask = smoothygrad < gradient_cutoff
    
    mask = np.logical_and(Partition_mask, gradient_mask)
    
    if np.sum(mask) > 100: #Need at least 100 eligible datapoints to continue
        fit_params = polynomial.polyfit(X[mask],Y[mask], order)
        # m, c, r_value, p_value, std_err = stats.linregress(X[mask],Y[mask])

        baseline = polynomial.polyval(X, fit_params)
        OffsetY = Y - baseline

        if np.mean(OffsetY[:10]) < 0:
            # If the start of the constant compliance region is less than zero, process has failed
            return None, None
        
        if np.any(bonus_ForceData) == None:
            return copy.deepcopy(np.array([X, OffsetY]))
        else:
            baseline = polynomial.polyval(bonus_ForceData[0], fit_params)
            bonus_OffsetY = bonus_ForceData[1] - baseline
            return copy.deepcopy(np.array([X, OffsetY])), copy.deepcopy(np.array([bonus_ForceData[0], bonus_OffsetY]))


    else:
        return None, None
    


def zeroForceCurves(ForceData):
    """
    Performs a zeroing on force vs. separation curves
    
    (NOT displacement / z-piezeo position curves - use "ConvertToForceVSep" to convert
    to force vs. separation first)
    """

    if ForceData.ndim == 3:
        for FD in ForceData:
            compliance = extractHardContactRegion(ForceData)
            comp_len_cutoff = int(compliance.shape[1]/1.5)
            FD[0] -= np.mean(compliance[0][:comp_len_cutoff])

    else:
        compliance = extractHardContactRegion(ForceData)
        comp_len_cutoff = int(compliance.shape[1]/1.5)
        ForceData[0] -= np.mean(compliance[0][:comp_len_cutoff])

    return ForceData



def calculateSensitivity(ForceData):
    compliance = extractHardContactRegion(ForceData)

    comp_len_cutoff = int(compliance.shape[1]/1.5)
    Sen = -1/stats.linregress(compliance[0, :comp_len_cutoff], compliance[1, :comp_len_cutoff]).slope

    return Sen



def ConvertToForceVSep(ForceData, sensitivity=None, spring_constant=1):
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
  
    
def find_SMpulloffs(ForceSepRetract, verbose=False, lowest_value=5):
    """
    Returns values in distance from susbtrate and adhesion force (nN) (as a positive quantity)
    
    """
    x, y = ForceSepRetract
    x = x[~np.isnan(y)]
    y = y[~np.isnan(y)]
    
    mask1 = x > lowest_value
    x = x[mask1]
    y = y[mask1]

    
    # Smooth the y-signal to remove outliers when masking
    window_length = int(len(x)/80)
    if window_length%2 == 0:
        window_length += 1
        
        
    newy = savgol_filter(y, window_length, 1)
    mask = newy < 0

    PO_x = x[mask]
    PO_y = y[mask]
    PO_newy = newy[mask]
    
    xdiff = np.abs(np.diff(PO_x))
    ydiff = np.diff(PO_newy)

    splits = np.argwhere(np.logical_or(xdiff>2*np.quantile(xdiff, 0.95),
                                   ydiff>2*np.quantile(ydiff, 0.95)))[:,0]
    splits = np.concatenate([np.array([0]), splits, np.array([len(xdiff)-5])])
    
    split_SM_curves = []
    discarded_SM_curves = []

    for [sx1, sx2] in zip(splits[:-1], splits[1:]):
        start_slice = sx1-10
        if start_slice < 0: # If start slice < 0 will stuff up indexing later
            start_slice = 0
        
        end_slice = sx2-3
        if end_slice > len(PO_x):
            end_slice = len(PO_x)-1
    
        tempx = PO_x[start_slice:end_slice].copy()
        tempy = PO_y[start_slice:end_slice].copy()
        
        if len(tempx) > 50: # If the array has less than 30 entries, then don't worry about it (keep in mind we're padding)
            
            # We only take values with a negative slope (overall), because we just want the
            # pull-off events.
            
            a, b = np.polyfit(tempx, tempy, 1)
            
            run = tempx[-1] - tempx[0]
            
            split_mask = np.logical_and(x>tempx[0], x<tempx[-1])
            if a*run > -0.01:
                discarded_SM_curves.append([abs(x[split_mask]), -y[split_mask]])
            elif PO_x[end_slice] + 10 > x[-1]:
                discarded_SM_curves.append([abs(x[split_mask]), -y[split_mask]])
            else:
                split_SM_curves.append([abs(x[split_mask]), -y[split_mask]])

                
    if verbose:
        return split_SM_curves, discarded_SM_curves, [[PO_x, PO_y, PO_newy]]
    else:
        return split_SM_curves
    
    
    
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



def resampleForceDataArray(ForceData):  
    maxZval = 0
    minZval = 0
    maxLen = 0

    for data in ForceData:
        z = data[0]
        localMin = min(z)
        localMax = max(z)

        if minZval > localMin:
            minZval = localMin

        if maxZval < localMax:
            maxZval = localMax

        if maxLen < len(z):
            maxLen = len(z)
            
    newZ = np.linspace(minZval, maxZval, num=maxLen)

    newForceData = []

    for data in ForceData:
        newForceData.append([newZ, np.interp(newZ, data[0], data[1], right=np.nan)])

    return np.array(newForceData)