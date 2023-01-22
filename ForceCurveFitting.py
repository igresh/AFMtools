import numpy as np
from scipy.signal import savgol_filter


def residual(params, x, data, model):
    model_out = model(x, params)
    return (data-model_out)/data

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
    xi = x/L
    return baseline + 1e9*(k*T/l)*(xi+1/(4*(1-xi)**2)-0.25)


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
