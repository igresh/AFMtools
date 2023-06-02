import numpy as np
from scipy.signal import savgol_filter
from ForceCurveFuncs import plotdebug
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


def variable_smoothing (x, y, initial_window, final_window, final_distance=50, increments=5):
    y = copy.copy(y)

    if x[-1] > final_distance:
        final_distance = x[-1]

    initial_distance = 0

    boundaries       = np.linspace(initial_distance, final_distance, increments+1)
    lower_boundaries = boundaries[0:-1]
    upper_boundaries = boundaries[1:]
    window_increments = np.linspace(initial_window, final_window, increments)



    for lb, ub, wi in zip(lower_boundaries, upper_boundaries, window_increments):


        if len(x[x>ub]) < 2*final_window:
            ub = x[-1]
        if lb > x[-1]:
            break


        # print (lb, ub)

        mask = np.logical_and(x>lb, x<ub)

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


def variable_ythreh (x_pos, base_thresh=0.0015, min_thresh=0.0005):
    """
    base pos - at 25 nm
    """
    thresh = base_thresh - (base_thresh-min_thresh)*(x_pos - 100)*0.1

    thresh[thresh<min_thresh] = min_thresh

    return thresh

def variable_lengththresh(xpos, start_x=30, tmin=1, tmax=30, cutoff=250):
    
    frac = (cutoff-xpos)/(cutoff-start_x)

    if frac < 0:
        frac = 0
    elif frac > 1:
        frac = 1

    return tmin*frac + tmax*(1-frac)

def find_SMpulloffs(ForceSepRetract, verbose=False, debug=False, lowest_value=5, smooth_window_nm=2.5, ythreshold=0.01, ycutoff=0.02):
    """
    Returns values in distance from susbtrate and adhesion force (nN) (as a positive quantity)


    """

    debugger = plotdebug(debug=debug)

    x, y = ForceSepRetract
    x = x[~np.isnan(y)]
    y = y[~np.isnan(y)]



    xrange = x[-1] - x[0]


    numdp_per_nm = len(x)/xrange

    mask1 = x > lowest_value


    x = x[mask1]
    y = y[mask1]
    gradient, b = np.polyfit(x, y, 1)


    # # Smooth the y-signal to remove outliers when masking
    # window_length = int(smooth_window_nm*numdp_per_nm)
    # if window_length%2 == 0:
    #     window_length += 1

    # newy = savgol_filter(y, window_length, 1)

    newy = variable_smoothing(x, y, initial_window=numdp_per_nm*smooth_window_nm, final_window=numdp_per_nm*5*smooth_window_nm)

    xhalf = np.max(x)/2
    min_ydffthreshold =  np.quantile(np.abs(np.diff(newy[x>xhalf][:-20])), 0.95)
    xdffthreshold =  2#np.quantile(np.abs(np.diff(x[x>xhalf])), 0.99)
    mask = newy < ycutoff

    xhalf_idx = int(len(x)/2)
    gradient_of_BL = (np.average(y[-20:]) - np.average(y[xhalf_idx:xhalf_idx+20]))/xhalf


    PO_x = x[mask]
    PO_y = y[mask]
    PO_newy = newy[mask]

    xdiff = savgol_filter(np.abs(PO_x), 3,1, deriv=1)
    ydiff = savgol_filter(PO_newy, 11,1, deriv=1)

    if len(PO_x) > 42:
        yddiff = savgol_filter(ydiff, 41,2, deriv=1)
    else:
        yddiff = np.zeros_like(PO_x, dtype=bool)
 
    # [[x, y],  PO_x, 10*ydiff], [PO_x, 1000*yddiff] 
    debugger.plot(curves=[[x, newy]], labels=['smooth', 'diff', 'ddiff'], color='k')

    # debugger.scatter(curves=[[PO_x[:-1][xdiff>xdffthreshold], PO_newy[:-1][xdiff>xdffthreshold]],
    #                       [PO_x[:-1][ydiff>ydffthreshold], PO_newy[:-1][ydiff>ydffthreshold]]], labels=['xdiff', 'ydiff'])

    xdifmask = xdiff>xdffthreshold
    ydifmask = ydiff>variable_ythreh(PO_x,  min_thresh=min_ydffthreshold)
    yddifmask = yddiff < -8e-5
    yddifmask[PO_x<15] = False

    splits = np.argwhere(np.any([xdifmask,ydifmask, yddifmask], axis=0))[:,0]

    # debugger.scatter([[PO_x[splits], PO_newy[splits]]], labels=['splits'])
    debugger.scatter([[PO_x[xdifmask], PO_newy[xdifmask]],
                      [PO_x[ydifmask], PO_newy[ydifmask]],
                      [PO_x[yddifmask], PO_newy[yddifmask]]], labels=['splits - xdif', 'splits - ydif', 'splits - yddif'], marker='x')

    splits = np.concatenate([np.array([0]), splits, np.array([len(xdiff)-5])])

    split_SM_curves = []
    discarded_SM_curves = []

    too_short = 0
    too_positive = 0
    too_small_delta = 0
    too_close_to_end = 0

    for [sx1, sx2] in zip(splits[:-1], splits[1:]):
        if gradient < -0.0001 and b < 1:
            if debug:
                print (f'killing it because of gradient: {gradient} {b}')
            break # bin it - too hard. (if the whole curve has a negative gradient every change in derivitive looks like a pull-off)


        start_slice = sx1
        if start_slice < 0: # If start slice < 0 will stuff up indexing later
            start_slice = 0

        end_slice = sx2-10
        if end_slice > len(PO_x):
            end_slice = len(PO_x)-1

        tempx = PO_x[start_slice:end_slice].copy()
        tempy = PO_y[start_slice:end_slice].copy()

        # if len(tempx) > min_dp:
            # window_length = int(len(tempx)/2)
            #if window_length%2 == 0:
             #   window_length += 1

            # gradient_mask_2 = savgol_filter(tempy, window_length, 1, deriv=1) < 0
            # tempx = tempx[gradient_mask_2]
            # tempy = tempy[gradient_mask_2]

        length = len(tempx)

        if length>3:
            # print (length)
            length_nm = (tempx[-1] - tempx[0])
            # print (length_nm , '/' , variable_lengththresh(tempx[0]) )

            if length_nm > variable_lengththresh(tempx[0]):

                # We only take values with a negative slope (overall), because we just want the
                # pull-off events.

                a, b = np.polyfit(tempx, tempy, 1)

                # print (a, -gradient_of_BL + 1e-5, a < (-gradient_of_BL + 1e-5))


                run = tempx[-1] - tempx[0]
                rise = np.mean(tempy[-int(length/10):-1]) - np.mean(tempy[0:int(length/10)])
                split_mask = np.logical_and(x>tempx[0], x<tempx[-1])

                if a > (-gradient_of_BL + 1e-5):
                    discarded_SM_curves.append([abs(x[split_mask]), -y[split_mask]])
                    too_positive += 1
                    debugger.plot(curves=[[tempx, tempy]], labels=['too positive'], color='r')


                elif rise > -ythreshold:
                    discarded_SM_curves.append([abs(x[split_mask]), -y[split_mask]])
                    too_small_delta += 1
                    debugger.plot(curves=[[tempx, tempy]], labels=['too small delta'], color='b')


                elif PO_x[end_slice] + 10 > x[-1]:
                    discarded_SM_curves.append([abs(x[split_mask]), -y[split_mask]])
                    too_close_to_end += 1
                else:
                    split_SM_curves.append([abs(x[split_mask]), -y[split_mask]])
            else:
                too_short += 1
                debugger.plot(curves=[[tempx, tempy]], labels=['too short'], color='xkcd:purple')

    if debug:
        print(f"too short: {too_short}\ntoo positive: {too_positive}\nsmall range: {too_small_delta}\ntoo close to end: {too_small_delta}")


    if verbose:
        return split_SM_curves, discarded_SM_curves, [[PO_x, PO_y, PO_newy]]
    else:
        return split_SM_curves
