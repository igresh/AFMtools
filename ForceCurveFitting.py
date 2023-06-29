import numpy as np
from scipy.signal import savgol_filter
from Utility import plotdebug
import copy


def residual(params, x, data, model):
    model_out = model(x, params)
    # squaring everything to weight values further from zero higher than those closer to zero
    return (data-model_out) # This should be /data , but that screws up when there are datapoints really close to zero

def WLC_model(x, params):
    """
    takes values in nm, returns values in nN
    """
    l = params['persistanceLength']
    L = params['contourLength']
    T = params['temperature']
    baseline = params['baseline']

    l *= 1e-9
    k = 1.380649e-23

    below_CL = np.zeros_like(x)
    above_CL = np.zeros_like(x)

    xi = x[x<L]/L
    below_CL[x<L] = 1e9*(k*T/l)*(xi+1/(4*(1-xi)**2)-0.25)
    xi = 0.999
    above_CL[x>=L] = 1e9*(k*T/l)*(xi+1/(4*(1-xi)**2)-0.25)
    return baseline + below_CL + above_CL


def variable_smoothing (x, y, initial_window, final_window, start_pos=25, end_pos=200, increments=3):
    y = copy.copy(y)

    if x[-1] > end_pos:
        end_pos = x[-1]

    initial_distance = 0

    boundaries       = np.linspace(initial_distance, end_pos, increments+1)
    boundaries[0]    = 0

    lower_boundaries = np.flip(boundaries[0:-1])
    upper_boundaries = np.flip(boundaries[1:])
    window_increments = np.flip(np.linspace(initial_window, final_window, increments))




    for lb, ub, wi in zip(lower_boundaries, upper_boundaries, window_increments):


        if len(x[x>ub]) < 2*final_window:
            ub = x[-1]
        # if lb > x[-1]:
        #     break


        mask = x>lb

        w = int(wi)
        if len(x[mask]) <= w + 1:
            w =  len(x[mask]) - 2

        w = int(w)
        if w%2 == 0:
            w += 1

        try:
            y[mask] = savgol_filter(y[mask], w, 1)
        except ValueError:
            print (w, len(y[mask]))
    return y


def variable_thresh(xpos, vstart=1, vend=40, start_pos=25, end_pos=200):
    
    frac = (end_pos-xpos)/(end_pos-start_pos)

    if frac < 0:
        frac = 0
    elif frac > 1:
        frac = 1

    return vstart*frac + vend*(1-frac)


def find_mask_centers(mask):
    """
    Find the centres of a group of masks in a masked array.
    """
    try:
        bounds = np.where(np.diff(mask))[0]
        centres = np.average([bounds[1:], bounds[:-1]], axis=0).astype(int) 
    except ZeroDivisionError:
        # print('find mask centres encountered Div 0')
        return mask

    if mask[0] == True:
        centres = centres[1::2]
    else:
        centres = centres[::2]

    new_mask = np.zeros_like(mask)
    new_mask[centres] = 1
    new_mask = new_mask.astype(bool)
    
    return new_mask

def find_SMpulloffs(ForceSepRetract, verbose=False, debug=False,
                    lowest_value_nm=5,
                    force_cutoff=0.02,
                    smooth_window_nm=[2.5, 12.5],
                    length_delta_threshold=[2,15],
                    force_delta_threshold=0.01,
                    low_dist = 50,
                    high_dist = 200,
                    force_gradient_cutoff=0.001,
                    min_dp = 15
                    ):
    """
    Returns values in distance from susbtrate and adhesion force (nN) (as a positive quantity)

    ForceSepRetract (np.array, required):
        2D numpy array, where the ForceSepRetract[0] is the seperation between the tip and the surface
        (in nanometers) and vForceSepRetrat[1] is the force acting on the cantilever (in nanonewtons).
    verbose (bool):
        If True, returns three arrays:
            split_SM_curves         - the accepted SM curves
            discarded_SM_curves     - the rejected SM curves
            [[PO_x, PO_y, PO_newy]] - masked x coords, masked y coords, masked smoothed y coords.
    debug (bool):
        If true, prints debugging output.
    lowest_value (float):
        The value where find_SMpulloffs will start looking for pull-off events. Typically
        around 5 nm.
    force_cutoff (float):
        The value below which force curves are considered possible. Assumes that the force curve has
        already been baseline corrected (i.e., baseline is at F=0 and flat).
    smooth_window_nm (float or list, default [2.5, 12.5]):
        The length (in nano meters) of the smoothing window used by savgol filter. This is converted to
        an integer (odd) number of datapoints before filtering. If it is a list, then it *must* have
        only two entries. The first entry will be the smoothing used before low_dist, the second entry
        will be the smoothing window used after high_dist. Values will vary linearly between low_dist
        and high_dist.
    length_delta_threshold (float or list, default [2, 15]):
        The minimum length (in nm) of a pull off event for it to be considered valid. If it is a list,
        then it *must* have only two entries. The first entry will be the threhold used before low_dist,
        the second entry will be the length used after high_dist. Values will vary linearly between
        low_dist and high_dist.
    force_delta_threshold (float, default=0.01):
        Minimum difference in force between first and last datapoint for the curve to be considered a single-
        molecule event. Mainly used as a de-noiser.
    low_dist (float):
        The distance where smooth_window_nm and length_delta_threshold begin varying (if a list is passed).
        Helpful when you want to find sharp, short features at low distance, but want to screen noise at
        longer distances.
    high_dist (float):
        The distance where smooth_window_nm and length_delta_threshold stop varying (if a list is passed).
        Helpful when you want to find sharp, short features at low distance, but want to screen noise at
        longer distances.
    force_gradient_cutoff (float):
        Gets rid of long bits that are technicall pull offs by all definitions but don't really look
        like them


    """
    # Start debugger - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    debugger = plotdebug(debug=debug)


    # Initialise variables - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    x, y = ForceSepRetract
    x = x[~np.isnan(y)]
    y = y[~np.isnan(y)]

    split_SM_curves = []
    discarded_SM_curves = []
    PO_x, PO_y, PO_newy=[], [], []



    if type(smooth_window_nm) is list:
        low_smooth_window_nm = smooth_window_nm[0]
        high_smooth_window_nm = smooth_window_nm[1]
    else:
        low_smooth_window_nm = smooth_window_nm
        high_smooth_window_nm = smooth_window_nm

    if type(length_delta_threshold) is list:
        low_length_delta_threshold = length_delta_threshold[0]
        high_length_delta_threshold = length_delta_threshold[1]
    else:
        low_length_delta_threshold = length_delta_threshold
        high_length_delta_threshold = length_delta_threshold

    xrange = x[-1] - x[0]
    numdp_per_nm = len(x)/xrange

    mask1 = x > lowest_value_nm
    x = x[mask1]
    y = y[mask1]

    xhalf = np.max(x)/4
    overall_gradient_mask = np.logical_and(np.abs(y) < 0.02, x>xhalf)
    overall_gradient_mask[-30:] = False
    overall_gradient, overall_intercept = np.polyfit(x[overall_gradient_mask], y[overall_gradient_mask], 1)


    newy = variable_smoothing(x, y,
                              initial_window=numdp_per_nm*low_smooth_window_nm,
                              final_window=numdp_per_nm*high_smooth_window_nm,
                              start_pos=low_dist, end_pos=high_dist)


    mask = newy < force_cutoff

    if np.abs(overall_gradient) > 0.00007:
        if debug:
            print (f'killing it because of gradient: {overall_gradient} {overall_intercept}')
        pass
    elif np.sum(mask) < 50:
        if debug:
            print (f'killing it because no points are below force_cutoff')

    else: # continue

        min_ydffthreshold =  2*np.quantile(np.abs(np.diff(newy[x>xhalf][:-20])), 0.98)
        xdffthreshold =  1#np.quantile(np.abs(np.diff(x[x>xhalf])), 0.99)


        PO_x = x[mask]
        PO_y = y[mask]
        PO_newy = newy[mask]

        xdiff = savgol_filter(np.abs(PO_x), 3,1, deriv=1)
        ydiff = savgol_filter(PO_newy, 11,1, deriv=1)

        if len(PO_x) > 42:
            yddiff = savgol_filter(ydiff, 41,2, deriv=1)
        else:
            yddiff = np.zeros_like(PO_x, dtype=bool)
     
        debugger.plot(curves=[[x, newy]], labels=['smooth', 'BL trend', 'ddiff'], color='k', zorder=10)
        debugger.plot(curves=[[x, x*overall_gradient+overall_intercept]], labels=['BL trend', 'ddiff'], color='gray',ls='--')
        debugger.plot(curves=[[x, x*0]], labels=['origin'], color='k',ls='--')


        xdifmask = xdiff>xdffthreshold
        ydifmask = ydiff>min_ydffthreshold

        yddifmask = np.abs(yddiff) > 1e-5
        yddifmask[PO_x<15] = False
        yddifmask = find_mask_centers (yddifmask)


        # key bit of code: split the curve up into regions that could be pull-off events - - - - - - - - - - - - - - - - - - -
        splits = np.argwhere(np.any([xdifmask,ydifmask, yddifmask], axis=0))[:,0]

        debugger.scatter([[PO_x[xdifmask], PO_newy[xdifmask]],
                          [PO_x[ydifmask], PO_newy[ydifmask]],
                          [PO_x[yddifmask], PO_newy[yddifmask]]], labels=['splits - xdif', 'splits - ydif', 'splits - yddif'], marker='x')

        splits = np.concatenate([np.array([0]), splits, np.array([len(xdiff)-5])]) # add a split to the start and end.


        # Initialize counting variables
        too_short = 0
        too_positive = 0
        too_small_delta = 0
        too_close_to_end = 0



        for [sx1, sx2] in zip(splits[:-1], splits[1:]):



            slice_length = sx2 - sx1
            if slice_length < min_dp:
                continue # this wont work - move on

            len2 = int(slice_length/50)
            len10 = int(slice_length/10)

            if len2 < 1:
                len2=1
            if len10 < 1:
                len10=1

            start_slice = sx1+len2
            if start_slice < 0: # If start slice < 0 will stuff up indexing later
                start_slice = 0

            end_slice = sx2-len2 # Cut off the last 2% of data points
            if end_slice > len(PO_x):
                end_slice = len(PO_x)-1

            tempx = PO_x[start_slice:end_slice].copy()
            tempy = PO_y[start_slice:end_slice].copy()
            tempsmoothy = PO_newy[start_slice:end_slice].copy()

            extramask = np.ones_like(tempx, dtype=bool)
            extramask[np.logical_and(tempx>(tempx[-1]-len10), savgol_filter(tempsmoothy,3,1, deriv=1)>0)] = False


            if np.sum(extramask) < min_dp:
                continue

            tempx = tempx[extramask]
            tempy = tempy[extramask]
            tempsmoothy = tempsmoothy[extramask]

            length_nm = (tempx[-1] - tempx[0])
            length = len(tempx)
            if length_nm > variable_thresh(tempx[0], vstart=low_length_delta_threshold,
                                           vend=high_length_delta_threshold, start_pos=low_dist, end_pos=high_dist):

                split_mask = np.logical_and(x>tempx[0], x<tempx[-1])

                gradient_PO, intercept_PO = np.polyfit(tempx, tempsmoothy, 1)

                start_loc = [tempx[0], np.mean(tempsmoothy[:len10])]
                end_loc  =  [tempx[-1], np.mean(tempsmoothy[-len10:-1])]
                peak = np.max(tempsmoothy)

                run = end_loc[0] - start_loc[0]
                rise = end_loc[1] - peak #start_loc[1]


                if gradient_PO > (overall_gradient + force_gradient_cutoff) or gradient_PO > (-force_gradient_cutoff):
                    discarded_SM_curves.append([abs(x[split_mask]), -y[split_mask]])
                    too_positive += 1
                    debugger.plot(curves=[[tempx, tempy]], labels=['too positive'], color='r', alpha=0.25)
                    debugger.plot(curves=[[tempx, tempx*gradient_PO+intercept_PO]], labels=['slope cutoff'], color='r')


                elif rise > -force_delta_threshold:
                    discarded_SM_curves.append([abs(x[split_mask]), -y[split_mask]])
                    too_small_delta += 1
                    debugger.plot(curves=[[tempx, tempy]], labels=['too small delta'], color='b', alpha=0.25)
                    debugger.scatter(curves=[start_loc, end_loc], labels=['start/end', 'start/end'], color='b')


                elif PO_x[end_slice] + 10 > x[-1]:
                    discarded_SM_curves.append([abs(x[split_mask]), -y[split_mask]])
                    too_close_to_end += 1
                else:
                    debugger.plot(curves=[[tempx, tempy]], labels=['accepted'], color='xkcd:green', alpha=0.5)
                    split_SM_curves.append([abs(x[split_mask]), -y[split_mask]])
            else:
                too_short += 1
                debugger.plot(curves=[[tempx, tempy]], labels=['too short'], color='xkcd:purple', alpha=0.25)

        if debug:
            print(f"too short: {too_short}\ntoo positive: {too_positive}\nsmall range: {too_small_delta}\ntoo close to end: {too_small_delta}")


    if verbose:
        return split_SM_curves, discarded_SM_curves, [[PO_x, PO_y, PO_newy]]
    else:
        return split_SM_curves
