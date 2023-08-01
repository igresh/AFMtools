# -*- coding: utf-8 -*-
"""
Author: Seamus Lilley (2023)

Script used to process the raw data obtained from the Asylum AFM and convert it into 
a meniscus force map. 
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('/Users/seamuslilley/Documents/GitHub/AFMtools')
import load_ardf  
import ForceCurveFuncs
from scipy.signal import savgol_filter 
from mpl_toolkits.mplot3d import Axes3D
#from pylab import * 
import scipy
from scipy.optimize import curve_fit
import math
from matplotlib import pyplot
from circle_fit import taubinSVD
from circle_fit import plot_data_circle
#%%
def data_load_in(file_name):
    """
    Imports the data and converts it from an .ARDF format into separate arrays of the 
    zsensor and deflection data. All of the experimental parameters are included in metadict.
    It will also flag if any of the deflection data has less than 200 points in it
    (indicating that there may have been something wrong with the forcecurve)

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
    [[raw, defl, delp], metadict] = load_ardf.ardf2hdf5(file_name+'.ARDF')
    #Converts them from lists to numpy arrays
    raw = np.array(raw, dtype = object)
    defl = np.array(defl,dtype = object)
    
    #Calculates the length of each of the deflection data
    fc_datapoints = np.zeros(len(defl))
    for i in range(len(defl)):
        fc_datapoints[i] = len(defl[i])
        if fc_datapoints[i] < 200:
            print('You have less than 200 data points for the '+str(i)+'th forcecurve')
    
    return raw, defl, metadict

def data_convert(raw, defl, metadict):
    """
    Takes the raw (zsensor and deflection) data and converts it into a force-separation format.
    The resolution of the map for x and y must be the same length

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
    ExtendsXY, RetractXY, ExtendsForce, RetractForce = ForceCurveFuncs.process_zpos_vs_defl(raw, defl,metadict,failed_curve_handling = 'retain')
    #Calculating the resolution of the plot 
    points_per_line = int(np.sqrt(len(ExtendsForce)))
    
    ExtendsForce = ForceCurveFuncs.resampleForceDataArray(ExtendsForce)
    ExtendsForce = np.reshape(ExtendsForce,(points_per_line,points_per_line,2,-1))
    
    #Rotates the points 180 deg to match the initial force map
    ExtendsForce = np.array(list(zip(*ExtendsForce[::-1])))
    ExtendsForce = np.array(list(zip(*ExtendsForce[::-1])))
    #Flips every second row to match the initial force map
    ExtendsForce[1::2, :] = ExtendsForce[1::2, ::-1]
    
    print('This is a '+ str(points_per_line) + ' resolution forcemap, over a'
          + metadict["ScanSize"] + ' area. This corresponds to ' + str(round(float(metadict["ScanSize"])/points_per_line,9)*1e9)
            + ' nm separation between pixels. The map was taken on '+date_taken)
    
    
    return(ExtendsForce,points_per_line)



def data_process(ExtendsForce,points_per_line):
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
    bubble_loc = np.zeros((points_per_line,points_per_line,3270))
    
    for i in range(points_per_line):
        for j in range(points_per_line):
            
            #Cleaning the x any y data (removing nan and smoothing the y)
            x,y = ExtendsForce[i][j]
            x = x[~np.isnan(y)]
            y = y[~np.isnan(y)]
            #y = y[x>0]
            #x = x[x>0]
            y = savgol_filter(y, 51, 2)
            
            #Differentiating the data and smoothing
            dy = np.diff(y)
            dy = savgol_filter(dy, 51, 2)

            
            dx = x[range(len(dy))]
            
            #Taking the second derivative and smoothing that
            d2y = np.diff(dy)
            d2y = savgol_filter(d2y,81,2)
            
            #Ensuring that the script identifies all spikes in gradient,
            #either positive or negative
            dy_abs = abs(dy)
            
            #Idenitifying the peaks in the (abs) first derivative
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
                    #Selecting the derivative values between two peaks
                    peak_diff_range = dx[peaks[k]:peaks[k+1]]
                    peak_range = y[peaks[k]:peaks[k+1]]
                    #Calculating how much of the values are positive
                    derivative_percent = sum(dy[peaks[k]:peaks[k+1]] > 0)/(peaks[k+1]-peaks[k])
                    
                    #Different cases considered for deciding if oil or gas and
                    #assigning it a generic placeholder
                    if derivative_percent > 0.7:
                        #print('Oil')
                        region_type[peaks[k]:peaks[k+1]] = 1
                    elif derivative_percent < 0.3:
                        #print('Gas')
                        region_type[peaks[k]:peaks[k+1]] = 2
                    elif abs(np.min(peak_range)) > 0.1e-8:
                        #print('Gas')
                        region_type[peaks[k]:peaks[k+1]] = 1
                    else:
                        #print('Oil')
                        region_type[peaks[k]:peaks[k+1]] = 2

                dropin_loc[i][j] = x[peaks[-1]]
            #Using the placeholders to find what height value corresponds to the 
            #top of the bubble and gas
            if sum(region_type == 2) != 0:
                bubble_loc = np.where(region_type == 2)
                if x[bubble_loc[0][-1]] > 9e-6: #Just avoiding the initial bit of the force curve
                    x[bubble_loc[0][-1]] = 0
                bubble_height[i][j] = x[bubble_loc[0][-1]]

            if sum(region_type == 1) != 0:
                oil_loc = np.where(region_type == 1)
                if x[oil_loc[0][-1]] > 9e-6: #Just avoiding the initial bit of the force curve
                    x[oil_loc[0][-1]] = 0
                oil_height[i][j] = x[oil_loc[0][-1]]

                

            
    return(dropin_loc,bubble_height, oil_height)

def heatmap2d(arr, file_name):
    """
    Plots the height array as a heatmap. Option to save the plot as well.
    
    Parameters
    ----------
    arr : np.ndarray
        A 2D array of heights calculated from each force curve.
    file_name : string, optional
        Name of the file, should be included if saving the figure   

    Returns
    -------
    None.

    """
    newpath_map = newpath + '/heatmap'
    if not os.path.exists(newpath_map):
        os.makedirs(newpath_map)
        
    image_size = int(float(metadict["ScanSize"])/1e-6)
    
    if np.sum(arr == bubble_height) == points_per_line**2:
        file_name += 'bubble_height_'
        colour = 'viridis'
        title = 'Bubble Height'
    elif np.sum(arr == oil_height) == points_per_line**2:
        file_name += 'oil_height_'
        colour = 'inferno'
        title = 'Oil Height'
    else:
        file_name += 'oil_height_'
    
    #Plotting the heatmap
    plt.figure()
    plt.imshow(arr, cmap=colour,extent=[0,image_size,0,image_size])
    cbar = plt.colorbar()
    cbar.set_label('Thickness ($\mu$m)', rotation = 270, labelpad = 20)
    plt.xlabel('x ($\mu$m)')
    plt.ylabel('y ($\mu$m)')
    plt.xticks(np.arange(0,image_size+1,image_size/4))
    plt.yticks(np.arange(0,image_size+1,image_size/4))
    plt.title(title)
    if save_heatmap == True:
        save_name = file_name+'heatmap'+'.png'
        plt.savefig(newpath_map + '/' + save_name)
    plt.rcParams['figure.dpi'] = 1000
    plt.show()
    
def forcemapplot(data,coords,f_name = ''):
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

    Returns
    -------
    None.

    """
    #Creating a subfolder in the data folder for the force curves
    newpath_forcecurve = newpath+'/forcecurve'
    if not os.path.exists(newpath_forcecurve):
        os.makedirs(newpath_forcecurve)
    
    #Unit conversions
    x = data[0]/1e-6 
    y = data[1]/1e-9 
    
    coord_x = int(coords[0])
    coord_y = int(coords[1])
    jump_in = dropin_loc[coord_x,coord_y]/1e-6
    bubble_h = bubble_height[coord_x,coord_y]/1e-6
    oil_h  = oil_height[coord_x,coord_y]/1e-6
    

    #print(height)
    
    coords = str(coords)
    
    #Plotting the force curve
    
    plt.figure()
    plt.plot(x,y,c='tab:blue')
    plt.xlabel('Displacement ($\mu$m)')
    plt.ylabel('Force (nN)') 
    plt.rcParams['figure.dpi'] = 500
    #plt.ylim((min(y)-10,150))
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.xlim((0,1.5*np.max(oil_height)*1e6))
    #plt.axvline(jump_in, c='tab:red', label = 'Initial Jump-in')
    plt.axvline(bubble_h, c='tab:green', label = 'Bubble Height')
    plt.axvline(oil_h, c='tab:orange', label = 'Oil Height')
    plt.legend()
    
    #Saving out the force curve
    if save_forcemap == True:
        save_name = file_name + coords + 'forcecurve'+'.png'
        plt.savefig(newpath_forcecurve + '/' + save_name)
    plt.show()
    
def is_MAC(file_name = '',date_taken = '',mac = True,initiator = False):
    """
    The specific folder path will change depending on if I (Seamus) am working
    on a Mac or PC. This helps to switch between the two. Not necessary if only
    working from one device. Update the path for different users.

    Parameters
    ----------
    mac : boolean, optional
        Are you (Seamus) working on your mac? The default is True.

    Returns
    -------
    newpath : str
        Defines the path that specifies what folder to work in. Will change
        depending on if I (Seamus) am working on a Mac or PC.

    """
    if mac == True:
        newpath = r'/Users/seamuslilley/Library/CloudStorage/OneDrive-Personal/University/USYD (2021-)/Honours/AFM Data Processing/'
    else:
        newpath = r'C:/Users/Seamu/OneDrive/University/USYD (2021-)/Honours/AFM Data Processing/' 
        
    os.chdir(newpath)
    if initiator == False:
        newpath = newpath + date_taken + '/' +file_name
    return newpath
    

def side_profile(heights,row,horizontal = True):
    """
    Plots the side profile of the data for a specified row (with the option
                                                            to instead select
                                                            a column)

    Parameters
    ----------
    heights : np.array
        An array of all of the height data for either the bubble or oil.
    row : int
        Specific row that you want to take a side profile of.
    horizontal : boolean, optional
        The option to take a vertical side profile instead. The default is True.

    Returns
    -------
    None.

    """
    
    newpath_sideprofile = newpath+'/sideprofile'
    if not os.path.exists(newpath_sideprofile):
        os.makedirs(newpath_sideprofile)
        
    if horizontal == True:
        oil_y = heights[0][row,:]/1e-6
        bubble_y = heights[1][row,:]/1e-6
    else:
        oil_y = heights[0][:,row]/1e-6
        bubble_y = heights[1][row,:]/1e-6
    x = np.linspace(0,int(float(metadict["ScanSize"])/1e-6),points_per_line)
    plt.figure()
    plt.plot(x,oil_y,'x-',c='tab:blue')
    plt.plot(x,bubble_y,'x-', c = 'tab:red')
    plt.rcParams['figure.dpi'] = 500
    plt.xlabel('x ($\mu$m)')
    plt.ylabel('Height ($\mu$m)') 
    plt.legend(('Oil Height','Bubble Height'))
    
    if save_sideprofile == True:
        save_name = file_name + 'side_profile'+str(row)+'.png'
        plt.savefig(newpath_sideprofile + '/' + save_name)
    plt.show()
    
#%%
file_name = "SiS2bUWH06"
is_MAC(initiator = True, mac = False)
metadict = load_ardf.metadict_output(file_name+'.ARDF')
date_taken = metadict["LastSaveForce"][-7:-1]
newpath = is_MAC(file_name,date_taken, False)

save_forcemap = False
save_heatmap = False
save_sideprofile = False

x_pos, y_pos = (0,0)
#%%
raw, defl, metadict = data_load_in(file_name)
ExtendsForce, points_per_line = data_convert(raw, defl, metadict)

#%%
dropin_loc, bubble_height, oil_height = data_process(ExtendsForce, points_per_line)

#%%
heatmap2d(bubble_height,file_name)
heatmap2d(oil_height,file_name)

#%%
forcemapplot(ExtendsForce[x_pos][y_pos],(x_pos,y_pos))

#%%
side_profile([oil_height,bubble_height],5)


#%%

def droplet_CA(droplet_height):
    x = np.linspace(0,int(float(metadict["ScanSize"])/1e-6),points_per_line)
    y = droplet_height
    x = x[y>0.01]
    y = y[y>0.01]
    
    pos_coords = np.transpose(np.array((x,y)))
    xc, yc, r, sigma = taubinSVD(pos_coords)
    
    plot_data_circle(pos_coords,xc,yc,r)
    
    x_circ = np.linspace(-xc,xc,10000)
    y_circ = yc + np.sqrt(r**2-(x_circ-xc)**2)
    y_circ = y_circ[~np.isnan(y_circ)]

    h = max(y_circ)
    
    theta = np.rad2deg(np.arccos(1-h/r))   
    
    return theta
    
theta = droplet_CA(oil_height[5])    
    











