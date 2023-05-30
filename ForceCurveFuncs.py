import numpy as np
from scipy import stats
from scipy.signal import savgol_filter
import copy
import warnings
from itertools import compress
from numpy.polynomial import polynomial
from Utility import consecutive
import matplotlib.pyplot as plt

"""
Note - will have to change Isaacs script to set:
    
number_of_curves_before_equil = 1
flatten_retract_with_approach = True
drop_deviant_compReg = True
abs_forcecrop = 0.4

"""


def process_zpos_vs_defl(zpos, defl, metadict=None,
                         defl_units='nm', zpos_units='nm',
                         zpos_negative=True, max_to_process=None,
                         number_of_curves_before_equil=0, 
                         override_involS=False, override_spring_constant=False,
                         flatten_retract_with_approach=False, drop_deviant_compReg=False,
                         debug=False, abs_forcecrop=False, failed_curve_handling='remove'):
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

    flatten_retract_with_approach (bool):
        Whether or not to subtract the approach curve from the retract curve to flatten it. Useful for single-molecule
        force spectroscopy (note: this isn't how this works at the moment)

    drop_deviant_compReg (bool):
        Whether or not to drop curves with a constant compliance region more than 2 std away from the mean. Useful for 
        ensuring you only have high quality data.

    debug (bool):
        If True, will display additional output usefull for debugging.

    abs_forcecrop (float):
        Whether to crop out forces above or below a certain threshold. Useful because for large forces you might exceed the
        constant compliance threshold of the cantilever.

    failed_curve_handling (string):
        How to handle curves that don't meet the 'is_data_sanitary' test. 
        - if 'remove' then failed curves are not processed
        - if 'replace_nan' then failed curves are replaced with an array of nans of the same length as the original
          force curve
        - if 'retain' then failed curves are appended to the array in whatever state they were when they failed

    """

    # initialise variables
    ExtendsXY = []
    RetractsXY = []
    ExtendsForce = []
    RetractsForce = []
    Esens = []
    Rsens = []

    discard_count = 0
    replace_count = 0
    dodgy_count   = 0

    if drop_deviant_compReg:
        assert failed_curve_handling == 'remove', 'If drop_deviant_compReg is True, failed_curve_handling must be "remove"'

    if debug:
        fig, ax = plt.subplots()
    else:
        ax = None

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
        # Set up parameters
        data_sanitary = True # flag that will be used to determine if data is problematic
        XYdata = np.array([np.nan])
        ExtendXY = np.array([np.nan])
        RetractXY = np.array([np.nan])
        ExtendForce = np.array([np.nan])
        RetractForce = np.array([np.nan])
        Esen = np.nan
        Rsen = np.nan

        if debug:
            ax.legend()
            fig.show()

            userinput = input("Do you want to continue?")
            if userinput.lower() != 'y':
                break
            ax.cla()


        data_sanitary = is_data_sanitary([z, d], data_sanitary=data_sanitary)

        if data_sanitary is True:
            d = convert_defl_to_mV(d, defl_units, invOLS)
            z = convert_zpos_to_nm(z, zpos_units, flip=zpos_negative)
            XYdata = np.array([z,d])
        elif data_sanitary is False:
            data_sanitary = 'Dead on arrival'


        # Remove dwell drift
        data_sanitary = is_data_sanitary([z, d], data_sanitary=data_sanitary)
        if data_sanitary is True:
            XYdata = remove_dwell_drift(XYdata)
        elif data_sanitary is False:
            data_sanitary = 'Failed on unit conversion'



        # Split data into approach and retract - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        data_sanitary = is_data_sanitary(XYdata, data_sanitary=data_sanitary)
        if data_sanitary is True:
            ExtendXY, RetractXY = splitExtendRetract(XYdata, flip=True)
            plotdebug(debug=debug, curves=[ExtendXY, RetractXY], labels=['Extend', 'Retract'], clear=False, ax=ax)
        elif data_sanitary is False:
            data_sanitary = 'Failed on dwell drift removal'



        # Remove baseline  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        data_sanitary = is_data_sanitary([ExtendXY, RetractXY], data_sanitary=data_sanitary)
        if data_sanitary is True:
            ExtendXY  = RemoveBaseline_nOrder(ExtendXY, order=1, approachFraction=0.4, debug=debug)
            RetractXY = RemoveBaseline_nOrder(RetractXY, order=1, approachFraction=0.4, debug=debug)
            plotdebug(debug=debug, curves=[ExtendXY, RetractXY], labels=['Extend', 'Retract'], clear=False, ax=ax)

        elif data_sanitary is False:
            data_sanitary = 'Failed on splitExtendRetract'



        # Zero force curves - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        data_sanitary = is_data_sanitary([ExtendXY, RetractXY], data_sanitary=data_sanitary)
        if data_sanitary is True:
            ExtendXY, RetractXY = zeroForceCurves(ExtendXY), zeroForceCurves(RetractXY)
            plotdebug(debug=debug, curves=[ExtendXY, RetractXY], labels=['Extend', 'Retract'], clear=False, ax=ax)
        elif data_sanitary is False:
            data_sanitary = 'Failed on first baseline correction'



        # Calculate sensitivity - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        data_sanitary = is_data_sanitary([ExtendXY, RetractXY], data_sanitary=data_sanitary)
        if data_sanitary is True:
            Esen, Rsen = calculateSensitivity(ExtendXY), calculateSensitivity(RetractXY)
        elif data_sanitary is False:
            data_sanitary = 'Failed on split and normalize'


        if data_sanitary is True:
            if np.isnan(Esen) or np.isnan(Rsen):
                data_sanitary = False



        # Convert to force vs. separation  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        if data_sanitary is True:
            average_sens = (Esen + Rsen)/2
            ExtendForce  = ConvertToForceVSep(ExtendXY, sensitivity=average_sens, spring_constant=float(spring_constant))
            RetractForce = ConvertToForceVSep(RetractXY, sensitivity=average_sens, spring_constant=float(spring_constant))
            plotdebug(debug=debug, curves=[ExtendForce, RetractForce], labels=['Extend force', 'Retract force'], clear=False, ax=ax)
        elif data_sanitary is False:
            data_sanitary = 'Failed to find constant compliance sensitivity'

        if np.any(ExtendForce) == None:
            print ('force vs sep' , data_sanitary, (number_of_curves_before_equil + idx))

        # Correct remaining baseline curvature   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        data_sanitary = is_data_sanitary([ExtendForce, RetractForce], data_sanitary=data_sanitary)
        if data_sanitary is True:
            if flatten_retract_with_approach is True:
                ExtendForce, RetractForce = RemoveBaseline_nOrder(ExtendForce, approachFraction=0.1, bonus_ForceData=RetractForce)
        elif data_sanitary is False:
            print (f"entry {number_of_curves_before_equil + idx} Failed on Force conversion")

        if np.any(ExtendForce) == None:
            print ('baseline curve' , data_sanitary, (number_of_curves_before_equil + idx), RetractForce)

        #Clean up data, one last time
        data_sanitary = is_data_sanitary([ExtendForce, RetractForce], data_sanitary=data_sanitary)

        if data_sanitary is True:
            if abs_forcecrop:
                ExtendForce, RetractForce = clean_forceData(ExtendForce, RetractForce, forcecrop=abs_forcecrop)
        elif data_sanitary is False:
            data_sanitary = 'Failed on final baseline curvature correction'



        data_sanitary = is_data_sanitary([ExtendForce, RetractForce], data_sanitary=data_sanitary)
        if data_sanitary is True:
            pass
        elif data_sanitary is False:
            data_sanitary = 'Failed on final cleanup'
            print(f"entry {number_of_curves_before_equil + idx}: {data_sanitary}")
        else:
            print(f"entry {number_of_curves_before_equil + idx}: {data_sanitary}")


        if data_sanitary is True:
            # Append processed data to lists
            ExtendsXY.append(ExtendXY)
            RetractsXY.append(RetractXY)
            ExtendsForce.append(ExtendForce)
            RetractsForce.append(RetractForce)
            Esens.append(Esen)
            Rsens.append(Rsen)
        else:
            if failed_curve_handling == 'replace_nan':
                nanarr = np.empty_like(z)
                nanarr[:] = numpy.nan

                ExtendsXY.append(nanarr)
                RetractsXY.append(nanarr)
                ExtendsForce.append(nanarr)
                RetractsForce.append(nanarr)
                replace_count += 1

            elif failed_curve_handling == 'retain':
                ExtendsXY.append(ExtendXY)
                RetractsXY.append(RetractXY)
                ExtendsForce.append(ExtendForce)
                RetractsForce.append(RetractForce)
                dodgy_count += 1

            elif failed_curve_handling == 'remove':
                discard_count += 1
                if debug:
                    print('Data discarded')

    if len(Esens) > 1:
        AvExSens = np.mean(Esens)
        StdExSens = np.std(Esens)
        AvRetSens = np.mean(Rsens)
        StdRetSens = np.std(Esens)

        if drop_deviant_compReg:
        # Get rid of data that deviates from the mean sensitivity
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
    else:
        AvExSens = np.nan
        StdExSens = np.nan
        AvRetSens = np.nan
        StdRetSens = np.nan
        print ('Sensitivity not calculated, as no curves passed "is_data"')


    # Print stuff that you might want to know
    print("Extend Sensitivity: " + str(AvExSens) + " nm/V")
    print("Retract Sensitivity: " + str(AvRetSens) + " nm/V")
    num_curves = len(zpos) - number_of_curves_before_equil
    if failed_curve_handling == 'remove':
        print (f'{discard_count}/{num_curves} curves did not meet criteria and were discarded' )
    elif failed_curve_handling == 'replace_nan':
        print (f'{replace_count}/{num_curves} curves were replaced with nan arrays')
    elif failed_curve_handling == 'retain':
        print (f'{dodgy_count}/{num_curves} curves that did not meet criteria were left in the dataset')

    return [ExtendsXY, RetractsXY, ExtendsForce, RetractsForce]


def plotdebug(ax, debug=False, curves=[None], labels=[None], clear=False):
    if debug:
        if clear:
            ax.cla()
        for c, l in zip(curves, labels):
            try:
                ax.plot(*c, label=l)
            except:
                print(f"plotdebug could not plot curve with label: {l}")

    return


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


def is_data_sanitary(data, data_sanitary=True):
    """
    Takes Force vs Separation data and determines if theres anything seriously wrong with it.

    data_sanitary allows for previously determined values of is_data_sanitary to the function.
    If data_sanitary is not True, then is_data_sanitary returns data_sanitary without performing
    any processing.
    """
    if data_sanitary is True:
        for datum in data:
            if np.any(datum) == None:
                return False
            elif np.any(np.isnan(datum)):
                return False
            elif np.ndim(datum) == 1 and datum.shape[0] < 100:
                return False
            elif np.ndim(datum) == 2 and datum.shape[1] < 100:
                return False
            else:
                return True

    else:
        return data_sanitary


def clean_forceData(ApproachForceData, RetractForceData, force_std_thresh=0.01, forcecrop=0.15):
    """
    'cleans up' approach and retract data. It does this by cropping out forces above a certain threshold
    and re-zeroing them.

    """
    mask = (np.abs(ApproachForceData[1])<forcecrop)
    newApproachForceData = ApproachForceData.T[mask].T
    newApproachForceData = zeroForceCurves(newApproachForceData)

    mask = (np.abs(RetractForceData[1])<forcecrop)
    newRetractForceData = RetractForceData.T[mask].T
    newRetractForceData = zeroForceCurves(newRetractForceData)


    return newApproachForceData, newRetractForceData


def old_clean_forceData(ApproachForceData, RetractForceData, force_std_thresh=0.01, forcecrop=0.15):
    """
    'cleans up' approach and retract data

    """
    mask = (np.abs(ApproachForceData[1])<forcecrop)
    newApproachForceData = ApproachForceData.T[mask].T
    newApproachForceData = zeroForceCurves(newApproachForceData)

    mask = (np.abs(RetractForceData[1])<forcecrop)
    newRetractForceData = RetractForceData.T[mask].T
    newRetractForceData = zeroForceCurves(newRetractForceData)

    mask = newApproachForceData[0] > 50
    std_non_compliance = 0.5*np.std(newApproachForceData[1][mask])

    if std_non_compliance > force_std_thresh: # large fluctuations in approach curves (which should not be there):
        return None, None
    else:
        return newApproachForceData, newRetractForceData




def splitExtendRetract(ForceData, flip=False):
    """
    ForceData : a 2xN array, where the [0] is the Z-piezo position and [1] is the deflection voltage

    flip : bool, if True, constant compliance region is on the left.
    """

    maxIndex  = np.argmax(ForceData[1])
    Extend    = ForceData[:,0:maxIndex]
    Retract   = np.flip(ForceData[:,maxIndex:], axis=1)

    if flip:
        Extend  = np.flip(Extend, axis=1)
        Retract = np.flip(Retract, axis=1)

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

    newraw = savgol_filter(raw, window_length, 1)
    newrawgrad = np.gradient(newraw)

    mask = np.abs(newrawgrad)>np.quantile(np.abs(newrawgrad), 0.5)/2

    return XYdata.T[mask].T



def RemoveBaseline_nOrder(ForceData, order=3, approachFraction=0.2, bonus_ForceData=None, debug=False):
    X, Y = ForceData
    Xrange = np.max(X) - np.min(X)
    Partition_mask = X > np.min(X) + Xrange*approachFraction
    gradient_mask = None
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
        if debug:
            print (f'Num in Partition_mask: {np.sum(Partition_mask)}, num in gradient_mask: {np.sum(gradient_mask)}')
        return None, None



def zeroForceCurves(ForceData):
    """
    Performs a zeroing on force vs. separation curves

    (NOT displacement / z-piezeo position curves - use "ConvertToForceVSep" to convert
    to force vs. separation first)
    """

    ForceData = np.array(ForceData) # Sanitise data input

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

    try:
        compliance = extractHardContactRegion(ForceData)

        comp_len_cutoff = int(compliance.shape[1]/1.5)
        Sen = -1/stats.linregress(compliance[0, :comp_len_cutoff], compliance[1, :comp_len_cutoff]).slope

        return Sen

    except:
        return np.nan



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
