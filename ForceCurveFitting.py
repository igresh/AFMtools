import numpy as np
from scipy.signal import savgol_filter


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


def find_SMpulloffs(ForceSepRetract, verbose=False, debug=False, lowest_value=5):
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


    newy = savgol_filter(y, window_length, 2)
    xhalf = np.max(x)/2
    ydffthreshold =  2*np.quantile(np.abs(np.diff(newy[x>xhalf])), 0.95)
    xdffthreshold =  2*np.quantile(np.abs(np.diff(x[x>xhalf])), 0.95)
    mask = newy < 0.02

    PO_x = x[mask]
    PO_y = y[mask]
    PO_newy = newy[mask]

    xdiff = np.abs(np.diff(PO_x))
    ydiff = np.diff(PO_newy)

    splits = np.argwhere(np.logical_or(xdiff>xdffthreshold,
                                       ydiff>ydffthreshold))[:,0]
    splits = np.concatenate([np.array([0]), splits, np.array([len(xdiff)-5])])

    split_SM_curves = []
    discarded_SM_curves = []

    too_short = 0
    too_positive = 0
    too_small_delta = 0
    too_close_to_end = 0


    for [sx1, sx2] in zip(splits[:-1], splits[1:]):
        start_slice = sx1
        if start_slice < 0: # If start slice < 0 will stuff up indexing later
            start_slice = 0

        end_slice = sx2-10
        if end_slice > len(PO_x):
            end_slice = len(PO_x)-1

        tempx = PO_x[start_slice:end_slice].copy()
        tempy = PO_y[start_slice:end_slice].copy()



        if len(tempx) > 30:
            window_length = int(len(tempx)/2)
            if window_length%2 == 0:
                window_length += 1

            gradient_mask_2 = savgol_filter(tempy, window_length, 2, deriv=1) < 0
            tempx = tempx[gradient_mask_2]
            tempy = tempy[gradient_mask_2]

        length = len(tempx)
        if length > 30: # If the array has less than 30 entries, then don't worry about it (keep in mind we're padding)

            # We only take values with a negative slope (overall), because we just want the
            # pull-off events.

            a, b = np.polyfit(tempx, tempy, 1)

            run = tempx[-1] - tempx[0]
            rise = np.mean(tempy[-int(length/10):-1]) - np.mean(tempy[0:int(length/10)])
            split_mask = np.logical_and(x>tempx[0], x<tempx[-1])
            if a > 0:
                discarded_SM_curves.append([abs(x[split_mask]), -y[split_mask]])
                too_positive += 1
            elif rise > -0.02:
                discarded_SM_curves.append([abs(x[split_mask]), -y[split_mask]])
                too_small_delta += 1
            elif PO_x[end_slice] + 10 > x[-1]:
                discarded_SM_curves.append([abs(x[split_mask]), -y[split_mask]])
                too_close_to_end += 1
            else:
                split_SM_curves.append([abs(x[split_mask]), -y[split_mask]])
        else:
            too_short += 1

    if debug:
        print(f"too short: {too_short}\ntoo positive: {too_positive}\nsmall range: {too_small_delta}\ntoo close to end: {too_small_delta}")


    if verbose:
        return split_SM_curves, discarded_SM_curves, [[PO_x, PO_y, PO_newy]]
    else:
        return split_SM_curves
