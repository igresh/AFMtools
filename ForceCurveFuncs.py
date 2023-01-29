import numpy as np
from scipy import stats
from scipy.signal import savgol_filter
import copy
import warnings
from itertools import compress
from numpy.polynomial import polynomial
from Utility import consecutive
import matplotlib.pyplot as plt


def process_zpos_vs_defl(zpos, defl, metadict=None, defl_units='nm', zpos_units='nm', zpos_negative=True, max_to_process=None,
                         number_of_curves_before_equil=1, override_involS=False, override_spring_constant=False, debug=False, abs_forcecrop=0.4):
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

    discard_count = len(zpos)

    if debug:
        fig, ax = plt.subplots()

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

        if metadict == None:
            original_spring_constant = 1
        elif 'SpringConstant' in metadict.keys():
            original_spring_constant = metadict['SpringConstant']
        else:
            original_spring_constant = 1

    else:
        if metadict == None:
            spring_constant = 1
        elif 'SpringConstant' in metadict.keys():
            spring_constant = float(metadict['SpringConstant'])
        else:
            spring_constant = 1

        original_spring_constant = spring_constant

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
        if debug:
            ax.legend()
            fig.show()

            userinput = input("Do you want to continue?")
            if userinput.lower() != 'y':
                break
            ax.cla()


        if not is_data_sanitary([z, d]):
            print (f"entry {number_of_curves_before_equil + idx} dead on arrival")
            continue


        # Convert units
        d = convert_defl_to_mV(d, defl_units, invOLS)
        z = convert_zpos_to_nm(z, zpos_units, flip=zpos_negative)
        XYdata = np.array([z,d])


        # Remove dwell drift
        XYdata = remove_dwell_drift(XYdata)
        if not is_data_sanitary(XYdata):
            print (f"entry {number_of_curves_before_equil + idx} Failed on dwell drift removal")
            continue


        # Split data into approach and retract
        ExtendXY, RetractXY = splitExtendRetract(XYdata)
        ExtendXY = np.flip(ExtendXY, axis=1)
        RetractXY = np.flip(RetractXY, axis=1)
        # ExtendXY = ExtendXY[:,20:-20]
        # RetractXY = RetractXY[:,20:-20]

        if not is_data_sanitary([ExtendXY, RetractXY]):
            print (f"entry {number_of_curves_before_equil + idx} Failed on splitExtendRetract")
            if debug:
                ax.plot(*ExtendXY, label='ExtendXY')
                ax.plot(*RetractXY, label='RetractXY')
            continue

        if debug:
                ax.plot(*ExtendXY, label='ExtendXY')
                ax.plot(*RetractXY, label='RetractXY')

        # Remove baseline  (change to norder=1 in the future)
        ExtendXY  = RemoveBaseline_nOrder(ExtendXY, order=1, approachFraction=0.4)
        RetractXY = RemoveBaseline_nOrder(RetractXY, order=1, approachFraction=0.4)
        if not is_data_sanitary([ExtendXY, RetractXY]):
            print (f"entry {number_of_curves_before_equil + idx} Failed on first baseline correction")
            continue

        if debug:
                ax.plot(*ExtendXY, label='ExtendXY')
                ax.plot(*RetractXY, label='RetractXY')

        ExtendXY, RetractXY = zeroForceCurves(ExtendXY), zeroForceCurves(RetractXY)
        if not is_data_sanitary([ExtendXY, RetractXY]):
            print (f"entry {number_of_curves_before_equil + idx} Failed on split and normalize")
            continue


        # Calculate sensitivity
        Esen = calculateSensitivity(ExtendXY)
        Rsen = calculateSensitivity(RetractXY)
        if np.isnan(Esen) or np.isnan(Rsen):
            print (f"entry {number_of_curves_before_equil + idx} Failed to find constant compliance sensitivity")
            continue

        if debug:
            ax.cla()
            ax.plot(*ExtendXY, label='Extend')
            ax.plot(*RetractXY, label='Retract')

        # Convert to force vs. separation
        average_sens = (Esen + Rsen)/2
        # old_EF, old_RF = ExtendForce, RetractForce
        ExtendForce  = ConvertToForceVSep(ExtendXY, sensitivity=average_sens, spring_constant=float(spring_constant))
        RetractForce = ConvertToForceVSep(RetractXY, sensitivity=average_sens, spring_constant=float(spring_constant))
        if not is_data_sanitary([ExtendForce, RetractForce]):
            print (f"entry {number_of_curves_before_equil + idx} Failed on Force conversion")

        if debug:
            ax.plot(*ExtendForce, label='Extend force')
            ax.plot(*RetractForce, label='Retract force')


        # Correct remaining baseline curvature
        ExtendForce, RetractForce = clean_forceData(ExtendForce, RetractForce, force_std_thresh=1, forcecrop=100)
        if not is_data_sanitary([ExtendForce, RetractForce]):
            print (f"entry {number_of_curves_before_equil + idx} Failed on first cleanup")
            continue

        if debug:
            ax.plot(*ExtendForce, label='Extend force, corrected')
            ax.plot(*RetractForce, label='Retract force, corrected')

        ExtendForce, RetractForce = RemoveBaseline_nOrder(ExtendForce, approachFraction=0.1, bonus_ForceData=RetractForce)
        if not is_data_sanitary([ExtendForce, RetractForce]):
            print (f"entry {number_of_curves_before_equil + idx} Failed on baseline curvature correction")
            continue


        # Clean up data, one last time
        ExtendForce, RetractForce = clean_forceData(ExtendForce, RetractForce, force_std_thresh=0.02, forcecrop=abs_forcecrop)
        if not is_data_sanitary([ExtendForce, RetractForce]):
            print (f"entry {number_of_curves_before_equil + idx} Failed on second cleanup")
            continue


        # Append processed data to lists
        ExtendsXY.append(ExtendXY)
        RetractsXY.append(RetractXY)
        ExtendsForce.append(ExtendForce)
        RetractsForce.append(RetractForce)
        Esens.append(Esen)
        Rsens.append(Rsen)

        discard_count -= 1 # One more force curve that wasn't discarded

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


def clean_forceData(ApproachForceData, RetractForceData, force_std_thresh=0.01, forcecrop=0.15):
    mask = (np.abs(ApproachForceData[1])<forcecrop)
    newApproachForceData = ApproachForceData.T[mask].T
    newApproachForceData = zeroForceCurves(newApproachForceData)

    mask = (np.abs(RetractForceData[1])<forcecrop)
    newRetractForceData = RetractForceData.T[mask].T
    newRetractForceData = zeroForceCurves(newRetractForceData)

    mask = newApproachForceData[0] > 5
    std_non_compliance = np.std(newApproachForceData[1][mask])

    if std_non_compliance > force_std_thresh: # large fluctuations in approach curves (which should not be there):
        return None, None
    else:
        return newApproachForceData, newRetractForceData



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



def RemoveBaseline_nOrder(ForceData, order=3, approachFraction=0.2, bonus_ForceData=None):
    X, Y = ForceData
    Xrange = np.max(X) - np.min(X)
    Partition_mask = X > np.min(X) + Xrange*approachFraction

    window_length = int(len(Y)/50)
    if window_length%2 == 0:
        window_length += 1

    smoothygrad = np.abs(savgol_filter(Y, window_length, 1, deriv=1))

    if np.any(bonus_ForceData) == None:
        gradient_cutoff = np.median(smoothygrad[Partition_mask])
        gradient_mask = smoothygrad < gradient_cutoff
        mask = np.logical_and(Partition_mask, gradient_mask)

    else:
        mask = Partition_mask

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
        print (f'Num in Partition_mask: {np.sum(Partition_mask)}, num in gradient_mask: {np.sum(gradient_mask)}')
        return None, None



def zeroForceCurves(ForceData):
    """
    Performs a zeroing on force vs. separation curves

    (NOT displacement / z-piezeo position curves - use "ConvertToForceVSep" to convert
    to force vs. separation first)
    """

    if ForceData.ndim == 3:
        for FD in ForceData:
            try:
                compliance = extractHardContactRegion(ForceData)
                comp_len_cutoff = int(compliance.shape[1]/1.5)
                FD[0] -= np.mean(compliance[0][:comp_len_cutoff])
            except IndexError:
                return None

    else:
        try:
            compliance = extractHardContactRegion(ForceData)
            comp_len_cutoff = int(compliance.shape[1]/1.5)
            ForceData[0] -= np.mean(compliance[0][:comp_len_cutoff])
        except IndexError:
            return None

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
