# -*- coding : utf -8 -*-
"""
Author : Seamus Lilley (2023)

Script used to process the raw data obtained from the Asylum AFM and convert it into
a meniscus force map .
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(r'L:\ljam8326 Asylum Research AFM\Infused Teflon Wrinkle Samples in Air\230721 Samples')
import load_ardf
import ForceCurveFuncs
from scipy.signal import savgol_filter
from mpl_toolkits.mplot3d import Axes3D
# from pylab import *
import scipy
from scipy.optimize import curve_fit
import math
from matplotlib import pyplot
from circle_fit import taubinSVD
from circle_fit import plot_data_circle
import glob
import pandas as pd
from scipy import asarray as ar, exp
#%%
def data_load_in (file_name):
    """
    Imports the data and converts it from an .ARDF format into separate arrays of the
    zsensor and deflection data. All of the experimental parameters are included in
    metadict.
    Credit to Dr. Isaac Gresham for writing the load_ardf.ardf2hdf5 function which
    converts the raw data.

    Parameters
    ----------
    file_name : string
    The name of the that is to be imported and processed.

    Returns
    -------
    raw : np.ndarray
    An numpy object array that contains all of the zsensor data for each force curve.
    defl : np.ndarray
    An numpy object array that contains all of the deflection data for each force
    curve.
    metadict : dict
    A dictionary storing all of the experimental parameters used to produce the force
    map.

    """
    # Loads in the data
    [[raw, defl, delp], metadict] = load_ardf.ardf2hdf5(file_name)
    # Converts them from lists to numpy arrays
    raw = np.array(raw, dtype = object)
    defl = np.array(defl, dtype = object)

    # Calculates the length of each of the deflection data
    fc_datapoints = np.zeros (len(defl))
    for i in range (len(defl)) :
        fc_datapoints [i] = len(defl[i])
        if fc_datapoints [i] < 200:
            print ('You have less than 200 data points for the ’+str ( i )+’th forcecurve')

    return raw, defl, metadict

def data_convert (raw, defl, metadict):
    """
    Takes the raw (zsensor and deflection) data and converts it into a force - separation
    format.
    The resolution of the map for x and y must be the same length

    Parameters
    ----------
    raw : np.ndarray
    An numpy object array that contains all of the zsensor data for each force curve.
    defl : np.ndarray
    An numpy object array that contains all of the deflection data for each force
    curve.
    metadict : dict
    A dictionary storing all of the experimental parameters used to produce the force
    map.

    Returns
    -------
    ExtendsForce : np.array
    A 4(!) dimensional array that contains the force and separation data for all
    points
    taken in the force map. This is shaped to replicate the force map grid as seen in
    the
    Asylum software when taking a map.
    points_per_line : int
    The resolution of the force map.

    """
    # Processing the data for both the extend and retract curves
    ExtendsXY, RetractXY, ExtendsForce, RetractForce, AvExSens, AvRetSens = ForceCurveFuncs.process_zpos_vs_defl(raw, defl, metadict, failed_curve_handling = 'retain')
    # Calculating the resolution of the plot
    points_per_line = int(np.sqrt(len(ExtendsForce)))

    ExtendsForce = ForceCurveFuncs.resampleForceDataArray(ExtendsForce)
    ExtendsForce = np.reshape(ExtendsForce, (points_per_line, points_per_line, 2, -1))

    # Rotates the points 180 deg to match the initial force map
    ExtendsForce = np.array(list(zip(*ExtendsForce[ : : -1])))
    ExtendsForce = np.array(list(zip(*ExtendsForce[ : : -1])))
    # Flips every second row to match the initial force map
    ExtendsForce [1 : : 2, : ] = ExtendsForce[1 : : 2, : : -1]

    date_taken = metadict["LastSaveForce"]    

    print ('This is a '+ str (points_per_line) + ' resolution forcemap , over a'
           + metadict [" ScanSize "] + ' area . This corresponds to ' + str(round(float(
    metadict[ " ScanSize " ]) / points_per_line, 9)*1e9)
              + ' nm separation between pixels . The map was taken on ' + date_taken)


    return (ExtendsForce, points_per_line, AvExSens, AvRetSens)



def data_process (ExtendsForce, points_per_line, metadict):
    """
    Identifies the heights corresponding to the initial pull -in event for the given
    threshold value .

    Parameters
    ----------
    ExtendsForce : np. array
    A 4(!) dimensional array that contains the force and separation data for all
    points
    taken in the force map. This is shaped to replicate the force map grid as seen in
    the Asylum software when taking a map.
    points_per_line : int
    The resolution of the force map.

    Returns
    -------
    dropin_loc : np.array
    The heights corresponding to the initial point identified by the threshold,
    shaped in an array to match the initial force map.
    bubble_height : np.array
    The heights corresponding to the initial point of the gas layer as
    identified by the threshold, shaped in an array to match the initial force map.
    oil_height : np.array
    The heights corresponding to the initial point of the oil layer as
    identified by the threshold, shaped in an array to match the initial force map.
    bubble_def : np.array
    The calculated spring constants of the layer identified as gas,
    shaped in an array to match the initial force map.
    """
    # Initialising the arrays
    dropin_loc = np.zeros((points_per_line, points_per_line))
    bubble_height = np.zeros((points_per_line, points_per_line))
    oil_height = np.zeros((points_per_line, points_per_line))
    oil_grad = np.zeros((points_per_line, points_per_line))
    gas_grad = np.zeros((points_per_line, points_per_line))
    bubble_def = np.zeros((points_per_line, points_per_line))
    bubble_loc = np.zeros((points_per_line, points_per_line, 3270))
    y_gastot = []
    y_oiltot = []
    for i in range (points_per_line):
        for j in range (points_per_line):

            # Cleaning the x any y data ( removing nan and smoothing the y)
            x, y = ExtendsForce [i][j]
            x = x [~ np.isnan(y)]
            y = y [~ np.isnan(y)]
            #y = y[x >0]
            #x = x[x >0]
            #y = savgol_filter (y, 51 , 2)

            # Differentiating the data and smoothing
            dy = np.diff(y)
            dy = savgol_filter(dy, 51,2)


            dx = x [range(len(dy))]

            # Taking the second derivative and smoothing that
            d2y = np.diff(dy)
            d2y = savgol_filter(d2y,81,2)

            # Ensuring that the script identifies all spikes in gradient ,
            # either positive or negative
            dy_abs = abs(dy)

            # Idenitifying the peaks in the ( abs ) first derivative
            peaks = scipy.signal.find_peaks (dy_abs, 1e-10, distance = 50, prominence = 1e-10)
            peaks = peaks[0]
            if np.argmin(dy) == 0:
                peaks = np.insert (peaks, 0, np.argmin(dy))
            region_type = np.zeros(len(y))

            if len (peaks) == 0:
                print ('Water')
                dropin_loc [i][j] = 0

            else :
                for k in range (len(peaks) - 1):
                    # Selecting the derivative values between two peaks
                    peak_diff_range = dx [peaks [k]:peaks[k + 1]]
                    peak_range = y[peaks [k]:peaks[k + 1]]
                    # Calculating how often the gradient is positive
                    dy = np.diff(y)
                    derivative_percent = sum (dy[peaks[k] : peaks[k + 1]] > 0 ) / (peaks [k+1] - peaks[k])

                    # Different cases considered for deciding if oil or gas and
                    # assigning it a generic placeholder
                    if derivative_percent > 0.7:
                        # Oil
                        region_type [ peaks [ k ] : peaks [k + 1]] = 1
                    elif derivative_percent < 0.3 :
                        # Gas
                        region_type [ peaks [ k ] : peaks [k + 1]] = 2
                    else :
                        # print ( ’Oil ’)
                        region_type [ peaks [ k ] : peaks [k + 1]] = 1

                dropin_loc [i][j] = x [peaks [-1]]
            # Using the placeholders to find what height value corresponds to the
            # top of the bubble and gas

            if sum (region_type == 1) != 0:
                oil_loc = np.where(region_type == 1)
                if x [oil_loc [0][-1]] > 9e-6 : # Just avoiding the initial bit of the force curve
                    x [oil_loc [0][-1]] = 0
                oil_height [i][j] = x [oil_loc[0][-1]]

                if dropin_loc [i][j] > 7.1e-8:
                    init_oil = np.where (peaks == oil_loc[0][-1]+1)
                    oil_grad_range = y [peaks[init_oil[0][0]-1] : peaks[init_oil[0][0]]]
                    oil_grad_domain = x [peaks[init_oil[0][0]-1] : peaks[init_oil[0][0]]]
                    if len (oil_grad_range) < 10:
                        oil_grad [i][j] = 0

                    elif len (oil_grad_range) < 40:
                        oil_grad_range = oil_grad_range [ 10:-10]
                        oil_grad_domain = oil_grad_domain [ 10:-10]
                    else :
                        oil_grad_range = oil_grad_range [20:-20]
                        oil_grad_domain = oil_grad_domain [20:-20]

                    yo1 = np.min(oil_grad_range)
                    yo2 = np.max(oil_grad_range)

                    xo1 = oil_grad_domain[np.where(oil_grad_range == np.min(oil_grad_range))[0][0]]
                    xo2 = oil_grad_domain[np.where(oil_grad_range == np.max(oil_grad_range))[0][0]]

                    oil_grad[i][j] = np.abs((yo2 - yo1) / (xo2 - xo1))

            if sum(region_type == 2) != 0:
                bubble_loc = np . where (region_type == 2)
                if x [bubble_loc [0][-1]] > 9e-6 : # Just avoiding the initial bit of the force curve
                    x [bubble_loc [0][-1]] = 0
                bubble_height [i][j] = x [bubble_loc[0][-1]]

                if dropin_loc[i][j] > 7.1e-8:
                    init_gas = np . where ( peaks == bubble_loc [ 0 ] [ - 1]+1)

                    gas_grad_range = y [peaks[init_gas[0][0]-1] : peaks[init_gas[0][0]]]
                    gas_grad_domain = x [peaks[init_gas[0][0]-1] : peaks[init_gas[0][0]]]
                    if len (gas_grad_range) < 10:
                        gas_grad [i][j] = 0
                    elif len (gas_grad_range) < 40:
                        gas_grad_range = gas_grad_range [10:-10]
                        gas_grad_domain = gas_grad_domain [10:-10]
                    else :
                        gas_grad_range = gas_grad_range [20:-20]
                        gas_grad_domain = gas_grad_domain [20:-20]

                    yg1 = np.min(gas_grad_range)
                    yg2 = np.max(gas_grad_range)

                    xg1 = gas_grad_domain[np.where(gas_grad_range == np.min(gas_grad_range))[0][0]]
                    xg2 = gas_grad_domain[np.where(gas_grad_range == np.max(gas_grad_range))[0][0]]

                    gas_grad[i][j] = (yg2 - yg1) / (xg2 - xg1)

                    if gas_grad [i][j] != 0:
                        k_cant = float( metadict['SpringConstant'] )
                        bubble_def [i][j] = gas_grad [i][j]*k_cant / (k_cant - gas_grad [i][j])


    return (dropin_loc , bubble_height , oil_height , bubble_def)

def heatmap2d (arr, file_name, metadict, points_per_line, bubble_height, oil_height, newpath='./', save_heatmap=True):
    """
    Plots the height array as a heatmap. Option to save the plot as well.

    Parameters
    ----------
    arr : np.array
    A 2D array of heights calculated from each force curve.
    file_name : string, optional
    Name of the file, should be included if saving the figure

    Returns
    -------
    None.

    """
    newpath_map = newpath + '/ heatmap'
    if not os . path . exists (newpath_map):
        os . makedirs (newpath_map)

    image_size = int(float(metadict['ScanSize']) /1e-6)

    if np . sum (arr == bubble_height) == points_per_line**2:
        file_name += 'bubble_height_'
        colour = 'viridis'
        title = 'Bubble Height'
    elif np . sum (arr == oil_height) == points_per_line**2:
        file_name += 'oil_height_'
        colour = 'inferno'
        title = 'Oil Height'
    else :
        file_name + 'oil_height_'

    # Plotting the heatmap
    plt.figure ()
    plt.imshow (arr*1e6, cmap=colour, extent = [0, image_size, 0, image_size])
    cbar = plt.colorbar ()
    cbar.set_label ('Height ($\ mu$m )', rotation = 270, labelpad = 20)
    plt.xlabel ('x ($\ mu$m )')
    plt.ylabel ('y ($\ mu$m )')
    plt.xticks (np.arange(0, image_size + 1, image_size / 4))
    plt.yticks (np.arange(0, image_size + 1, image_size / 4))
    # plt . text (0.05 ,2.3 , ’a) ’, transform = ax1 . transAxes , fontsize = 14)
    # plt . title ( title )
    if save_heatmap == True :
        save_name = file_name + 'heatmap' + '.png'
        plt.savefig (newpath_map + '/' + save_name)
    plt.rcParams ['figure . dpi'] = 1000
    plt.show ()

def side_profile (heights, row, file_name, metadict, points_per_line, newpath = './', horizontal = True, save_sideprofile = True):
    """
    Plots the side profile of the data for a specified row ( with the option
                                                            to instead select
                                                            a column )

    Parameters
    ----------
    heights : np.array
        An array of all of the height data for either the bubble or oil.
    row : int
        Specific row that you want to take a side profile of.
    horizontal : boolean , optional
        The option to take a vertical side profile instead. The default is True.

    Returns
    -------
    None.

    """

    newpath_sideprofile = newpath + '/ sideprofile'
    if not os.path.exists (newpath_sideprofile):
        os.makedirs (newpath_sideprofile)

    if horizontal == True :
        oil_y = heights [0] [row, :] / 1e-6
        bubble_y = heights [1] [row, :] / 1e-6
    else :
        oil_y = heights [0][:, row] / 1e-6
        bubble_y = heights [1] [row, :] / 1e-6
    x = np.linspace (0, int(float(metadict [" ScanSize "]) /1e-6 ) , points_per_line)
    plt . figure ()
    # plt . plot (x, bubble_y ,’x ’, c = ’tab : green ’,label = ’Gas Height ’)
    plt . plot (x, oil_y, 'x', c = 'tab:orange', label = 'Oil Height')
    # plt . plot ( x_circ , y_circ , c = ’tab : red ’, label = ’Curve Fit ’)
    plt . rcParams ['figure . dpi'] = 500
    plt . xlabel ('x ($\ mu$m )')
    plt . ylabel ('Height ($\ mu$m )')
    plt . xlim ((0, 4))
    plt . ylim ((0, max(oil_y)))
    plt . legend ()

    if save_sideprofile == True :
        save_name = file_name + 'side_profile' + str(row) + '.png'
        plt.savefig (newpath_sideprofile + '/' + save_name)
    plt.show ()

def droplet_CA (droplet_height, row, metadict, points_per_line):
    """
    Calculates the contact angle for a given row using a circular fit of the data

    Parameters
    ----------
    droplet_height : np.array
        A 2D array of heights calculated from each force curve.
    row : TYPE
        The row used to calculate the contact angle.

    Returns
    -------
    theta : float
        The measured contact angle as determined by the fit.

    """
    x = np.linspace (0, int(float(metadict [ " ScanSize " ]) /1e-6 ), points_per_line)
    y = droplet_height[row]*1e6
    x = x [y >0.1]
    y = y [y >0.1]

    pos_coords = np.transpose(np.array((x , y)))
    xc, yc, r, sigma = taubinSVD(pos_coords)

    # plot_data_circle ( pos_coords ,xc ,yc ,r)

    x_circ = np . linspace (-xc, xc, 10000)
    y_circ = yc + np . sqrt (r**2 - (x_circ - xc)**2)
    x_circ = x_circ [~np.isnan(y_circ)]
    y_circ = y_circ [~np.isnan(y_circ)]

    h = max(y_circ)

    theta = np.rad2deg(np.arccos(1 - h/r))

    return theta

def forcemapplot (data, coords, file_name, dropin_loc, bubble_height, oil_height, f_name = '', peak = [], newpath = './', save_forcemap = True):
    """
    Plots a single force curve. Helps with debugging, as you only consider one
    force curve at a time. It also converts both of the parameters to nN and um.

    Parameters
    ----------
    data : np.array
    A single force curve.
    coords : Tuple
    The coordinates that correspond to the specific force curve that is being plotted. This will get
    included in the naming of the file if it is saved.
    file_name : string, optional
    Name of the file, should be included if saving the figure

    Returns
    -------
    None.

    """
    # Creating a subfolder in the data folder for the force curves
    newpath_forcecurve = newpath + '/ forcecurve'
    if not os.path.exists(newpath_forcecurve):
        os.makedirs(newpath_forcecurve)

    # Unit conversions
    x = data [0] / 1e-6
    y = data [1] / 1e-9

    coord_x = int (coords[0])
    coord_y = int (coords[1])
    jump_in = dropin_loc [coord_x, coord_y] / 1e-6
    bubble_h = bubble_height [coord_x, coord_y] / 1e-6
    oil_h = oil_height [coord_x, coord_y] / 1e-6

    surface_feature = pd.DataFrame(np.reshape(dropin_loc [dropin_loc > 1e-7], (1, -1)))
    feature_quantiles = pd.DataFrame.to_numpy(surface_feature.quantile([ 0.25, 0.5, 0.9], axis = 1))
    # print ( height )

    coords = str (coords)

    # Plotting the force curve

    plt.figure()
    plt.plot(x, y, c = 'tab : blue', label = '')
    plt.xlabel('Separation ($\ mu$m )')
    plt.ylabel('Force (nN)')
    plt.rcParams['figure.dpi'] = 500
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.xlim((-0.1, 1.2*feature_quantiles [2]*1e6)) # Scales the plot limit based onquantiles, not max values
    plt.axvline(jump_in, c = 'tab : red', label = 'Initial Jump -in')
    plt.axvline(bubble_h, c = 'tab : green', label = 'Gas Height')
    plt.axvline(oil_h, c = 'tab : orange', label = 'Oil Height')
    plt.legend()
    # Saving out the force curve
    if save_forcemap == True :
        save_name = file_name + coords + 'forcecurve' + '.png'
        plt.savefig(newpath_forcecurve + '/' + save_name)
    plt.show()

def interface_measurement (bubble_deflection, upper_bound, lower_bound = 0):
    """
    Determines the average spring constant, and associated error,
    by applying a Gaussian distribution

    Parameters
    ----------
    bubble_deflection : np.array
        A 2D array of the measured spring constants for the identified gas layer.
    upper_bound : float
        The largest spring constant included in the fit (N/m).
    lower_bound : float, optional
        The lowest spring constant included in the fit (N/m). The default is 0.

    Returns
    -------
    None.

    """
    bubble_deflection = bubble_deflection [np.logical_and(bubble_deflection > lower_bound,
                                                              bubble_deflection < upper_bound)]
    print(len(bubble_deflection))
    y, x = np.histogram(bubble_deflection, round(np.sqrt(len(bubble_deflection))))
    for i in x:
        i += ( x[1] - x[0])/2
    x = x [ : -1]
    n = len(x)
    mean = sum(x*y)/n
    sigma = sum(y*(x - mean)**2 )/n

    def gaus (x, a, x0, sigma):
        return a*np.exp(-(x - x0)**2 / (2*sigma**2))

    popt , pcov = curve_fit (gaus, x, y, p0=[max (y), 0.05, 0.01])

    x_gaus = x
    y_gaus = gaus (x, *popt)

    plt.hist(bubble_deflection, round(np.sqrt(len(bubble_deflection))))
    plt.plot(x, gaus (x, *popt ), '--' , label = 'Gaussian Fit')
    plt.xlabel('Bubble Spring Constant (N/m)')
    plt.ylabel('Number of Force Curves')
    plt.legend()
    plt.show ()
    print (popt [1:4])
    return popt [1:4]