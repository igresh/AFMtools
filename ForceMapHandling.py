# -*- coding: utf-8 -*-
"""
Author: Seamus Lilley (2023)

Script used to process the raw data obtained from the Asylum AFM and convert it into 
a meniscus force map. 
"""

"""
Importing all of the necessary packages

load_ardf: A separate set of functions written by Isaac Gresham to import the .ARDF files
ForceCurveFuncs: Another set of functions written by Isaac Gresham to convert the
                 imported .ARDF files into force-separation curves
savgol_filter: a smoothing function that is necessary for the force curves (reduces impact of outliers)
"""

import load_ardf  
import numpy as np
import matplotlib.pyplot as plt
import ForceCurveFuncs
from scipy.signal import savgol_filter 
import scipy

#%%
def data_load_in(file_name):
    """
    Imports the data and converts it from an .ARDF format into separate arrays of the 
    zsensor and deflection data. All of the experimental parameters are included in metadict

    Parameters
    ----------
    file_name : string
        The name of the that is to be imported and processed.

    Returns
    -------
    raw : np.ndarray
        An numpy object array that contains all of the zsensor data for each force curve.
    defl : np.ndarray
        An numpy object array that contains all of the deflection data for each force curve.
    metadict : dict
        A dictionary storing all of the experimental parameters used to produce the force map.

    """
    #Loads in the data
    [[raw, defl, delp], metadict] = load_ardf.ardf2hdf5(file_name)
    #Converts them from lists to numpy arrays
    raw = np.array(raw, dtype = object)
    defl = np.array(defl,dtype = object)
    
    return raw, defl, metadict

def data_convert(raw, defl, metadict):
    """
    Takes the raw (zsensor and deflection) data and converts it into a force-separation format.
    (!) The resolution of the map for x and y must be the same length

    Parameters
    ----------
    raw : np.ndarray
        An numpy object array that contains all of the zsensor data for each force curve.
    defl : np.ndarray
        An numpy object array that contains all of the deflection data for each force curve.
    metadict : dict
        A dictionary storing all of the experimental parameters used to produce the force map.

    Returns
    -------
    ExtendsForce : np.array
        A 4(!) dimensional array that contains the force and separation data for all points
        taken in the force map. This is shaped to replicate the force map grid as seen in the
        Asylum software when taking a map. 
    points_per_line : int
        The resolution of the force map.

    """
    #Processing the data for both the extend and retract curves
    ExtendsXY, RetractXY, ExtendsForce, RetractForce = ForceCurveFuncs.process_zpos_vs_defl(raw, defl,metadict)
    #Calculating the resolution of the plot 
    points_per_line = int(np.sqrt(len(ExtendsForce)))
    
    ExtendsForce = ForceCurveFuncs.resampleForceDataArray(ExtendsForce)
    ExtendsForce = np.reshape(ExtendsForce,(points_per_line,points_per_line,2,-1))
    
    #Rotates the points 180 deg to match the initial force map
    ExtendsForce = np.array(list(zip(*ExtendsForce[::-1])))
    ExtendsForce = np.array(list(zip(*ExtendsForce[::-1])))
    #Flips every second row to match the initial force map
    ExtendsForce[1::2, :] = ExtendsForce[1::2, ::-1]
    return(ExtendsForce,points_per_line)

def FindThreshold(data, nThresh, percent): 
    """
    Finds a relevant threshold based on the first 100 data points. If there are less
    that 100 points of data in the force curve, then the threshold is set to a number much 
    larger than typical values. This should then yield an error. 
    
    Parameters
    ----------
    data : np.ndarray
        The approach force curve for each of the force map measurements.
    nThresh : TYPE
        Number of standard of deviations away from the mean to set the threshold. This should
        be changed if the pull-in events are not being detected. 
    percent : int
        Percentage of the data that will be used when calculating the background average.

    Returns
    -------
    threshold : float
        A large enough change in force (or displacement) to indicate a pull-in 
        event.

    """
    #Initialising the diff array
    diff = []
    #Applying the less than 100 data points condition
    if np.count_nonzero(data) <= 100:               
      threshold = 1e-6                               
    else:
        #the difference between values for some percent of the data
        diff = abs(np.diff(data[0:int(percent*0.01*len(data))])) 
        average = np.mean(diff)
        std = np.std(diff)
        threshold = abs(average + nThresh*std)                                                                                                 
    return(threshold)  

def data_process(ExtendsForce,points_per_line,nThresh,percent):
    """
    Identifies the heights corresponding to the initial pull-in event for the given
    threshold value. 

    Parameters
    ----------
    ExtendsForce : np.array
        A 4(!) dimensional array that contains the force and separation data for all points
        taken in the force map. This is shaped to replicate the force map grid as seen in the
        Asylum software when taking a map. 
    points_per_line : int
        The resolution of the force map.

    Returns
    -------
    dropin_loc : np.ndarray
        The heights corresponding to the initial point identified by the threshold,
        shaped in an array to match the initial force map. 

    """
    #Initialising the arrays
    dropin_loc = np.zeros((points_per_line,points_per_line))
    bubble_height = np.zeros((points_per_line,points_per_line))
    oil_height = np.zeros((points_per_line,points_per_line))
    
    for i in range(points_per_line):
        for j in range(points_per_line):
            
            x,y = ExtendsForce[i][j]
            x = x[~np.isnan(y)]
            y = y[~np.isnan(y)]
            #y = y[x>0]
            #x = x[x>0]
            y = savgol_filter(y, 51, 2)
            
            
            dy = np.diff(y)
            dy = savgol_filter(dy, 51, 2)

            dx = x[range(len(dy))]
            
            d2y = np.diff(dy)
            d2y = savgol_filter(d2y,81,2)

            
            dx = x[range(len(dy))]
            
            dy_abs = abs(dy)
            
            peaks = scipy.signal.find_peaks(dy_abs,1e-10, distance = 50, prominence = 1e-10)
            peaks = peaks[0]
            if np.argmin(dy) == 0:
                peaks = np.insert(peaks,0,np.argmin(dy))
            region_type = np.zeros(len(y))

            if len(peaks) == 0:
                print('Water')
                dropin_loc[i][j] = 0

            else:
                for k in range(len(peaks)-1):
                    peak_diff_range = dx[peaks[k]:peaks[k+1]]
                    peak_range = y[peaks[k]:peaks[k+1]]
                    derivative_percent = sum(dy[peaks[k]:peaks[k+1]] > 0)/(peaks[k+1]-peaks[k])
                    
                    if derivative_percent == 1:
                        #print('Oil')
                        region_type[peaks[k]:peaks[k+1]] = 1
                    elif abs(np.min(peak_range)) > 0.6e-7:
                        #print('Oil')
                        region_type[peaks[k]:peaks[k+1]] = 1
                    elif derivative_percent < 0.6:
                        #print('Gas')
                        region_type[peaks[k]:peaks[k+1]] = 2
                    else:
                        #print('Gas')
                        region_type[peaks[k]:peaks[k+1]] = 2
            
                dropin_loc[i][j] = x[peaks[-1]]
            if sum(region_type == 2) != 0:
                bubble_loc = np.where(region_type == 2)
                bubble_height[i][j] = x[bubble_loc[0][-1]]
                
            if sum(region_type == 1) != 0:
                oil_loc = np.where(region_type == 1)
                oil_height[i][j] = x[oil_loc[0][-1]]
            
    return(dropin_loc,bubble_height, oil_height)

def heatmap2d(arr, image_size, f_name = '', save = False):
    """
    Plots the height array as a heatmap. Option to save the plot as well.
    
    Parameters
    ----------
    arr : np.ndarray
        A 2D array of heights calculated from each force curve.
    image_size : Tuple
        Size of the image (x,y) in um. This will rescale the x and y range from points
        to distance.
    file_name : string, optional
        Name of the file, should be included if saving the figure   
    save : bool, optional
        Gives you the option to save out the heatmap. The default is False.

    Returns
    -------
    None.

    """
    #Plotting the heatmap
    plt.figure()
    plt.imshow(arr, cmap='viridis',extent=[0,image_size,0,image_size])
    cbar = plt.colorbar()
    cbar.set_label('Height ($\mu$m)', rotation = 270, labelpad = 20)
    plt.xlabel('x ($\mu$m)')
    plt.ylabel('y ($\mu$m)')
    plt.xticks(np.arange(0,image_size+1,image_size/5))
    plt.yticks(np.arange(0,image_size+1,image_size/5))
    if save == True:
        f_name = f_name.replace('.ARDF','')
        save_name = f_name+'heatmap'+'.png'
        plt.savefig(save_name)
    plt.rcParams['figure.dpi'] = 1000
    plt.show()
    
def forcemapplot(data,coords,f_name = '', save = False):
    """
    Plots a single force curve. Helps with debugging, as you only consider one
    force curve at a time. It also converts both of the parameters to nN and um. 

    Parameters
    ----------
    data : np.ndarray
        A single force curve.
    coords : Tuple
        The coordinates that correspond to the specific force curve that is being plotted. This will get 
        included in the naming of the file if it is saved. 
    file_name : string, optional
        Name of the file, should be included if saving the figure 
    save : TYPE, optional
        Gives you the option of saving the force curve plot. The default is False.

    Returns
    -------
    None.

    """
    #Unit conversions
    x = data[0]#/1e-6 
    y = data[1]#/1e-9 
    
    coord_x = int(coords[0])
    coord_y = int(coords[1])
    jump_in = dropin_loc[coord_x,coord_y]
    bubble_h = bubble_height[coord_x,coord_y]
    oil_h  = oil_height[coord_x,coord_y]
    #print(height)
    
    coords = str(coords)
    
    #Plotting the force curve
    
    plt.figure()
    plt.plot(x,y,c='tab:blue')
    plt.xlabel('Displacement ($\mu$m)')
    plt.ylabel('Force (nN)') 
    plt.rcParams['figure.dpi'] = 500

    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.axvline(jump_in, c='tab:red', label = 'Initial Jump-in')
    plt.axvline(bubble_h, c='tab:green', label = 'Bubble Height')
    plt.axvline(oil_h, c='tab:orange', label = 'Oil Height')
    plt.legend()
    

    if save == True:
        f_name = f_name.replace(".ARDF","")
        f_name = f_name + coords + 'forcecurve'+'.png'
        plt.savefig(f_name)
    plt.show()


