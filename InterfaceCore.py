# -*- coding: utf-8 -*-
"""
Author: Seamus Lilley (2023)

Script used to process the raw data obtained from the Asylum AFM and convert it into 
a meniscus force map. 
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import load_ardf  
import ForceCurveFuncs
from Utility import plotdebug
from scipy.signal import savgol_filter 
#from pylab import * 
import scipy


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

def data_convert(raw, defl, metadict, zero_constant_compliance=True,
                 rotate_map_90=0):
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
        
    rotate_map : bool
        Whether to rotate force map by 90˚ * rotate_map.  May help to to match image
        up to that seen in other software packages.

    Returns
    -------
    ForceMap : np.array
        A 4(!) dimensional array that contains the force and separation data for all points
        taken in the force map. This is shaped to replicate the force map grid as seen in the
        Asylum software when taking a map. 
    
    ExtendsForce : np.array
        The plain old un-mappified array of force curves, just in case you wanted to
        plot them.
    points_per_line : int
        The resolution of the force map.

    """
    


    date_taken = metadict["LastSaveForce"][-7:-1]

    #Processing the data for both the extend and retract curves
    ExtendsXY, RetractXY, ExtendsForce, RetractForce = ForceCurveFuncs.process_zpos_vs_defl(raw, defl, metadict,
                                                                                            failed_curve_handling = 'retain',
                                                                                            zero_at_constant_compliance=zero_constant_compliance)
    
    # resample so all force curves are the same length
    ExtendsForce = ForceCurveFuncs.resampleForceDataArray(ExtendsForce)

    
    # Calculating the resolution of the plot 
    points_per_line = int(np.sqrt(ExtendsForce.shape[0]))
    ForceMap = np.reshape(ExtendsForce,(points_per_line,points_per_line,2,-1))
    
    # Flips every second row to match the initial force map
    ForceMap[1::2, :] = ForceMap[1::2, ::-1]


    for i in range(rotate_map_90):
        ForceMap = np.array(list(zip(*ForceMap[::-1])))
        
    print('This is a '+ str(points_per_line) + ' resolution forcemap, over a'
          + metadict["ScanSize"] + ' area.')
    print('This corresponds to ' + str(round(float(metadict["ScanSize"])/points_per_line,9)*1e9)
            + ' nm separation between pixels.')
    print('The map was taken on ' + date_taken)
    
    
    return(ForceMap, ExtendsForce, points_per_line)



def data_process(ExtendsForce,points_per_line, debug=False):
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
    
    
    debugplotter = plotdebug(debug=debug)


    #Initialising the arrays
    dropin_loc = np.zeros((points_per_line,points_per_line))
    bubble_height = np.zeros((points_per_line,points_per_line))
    oil_height = np.zeros((points_per_line,points_per_line))
    bubble_loc = np.zeros((points_per_line,points_per_line,3270))
    topog = np.zeros((points_per_line,points_per_line))
    
    Oil = 1
    Gas = 2

    for i in range(points_per_line):
        for j in range(points_per_line):
            
            # Debugging stuff:
            if debug==True and debugplotter.plotted == True:
                debugplotter.show_plot()
                userinput = input("Do you want to continue?")
                if userinput.lower() != 'y':
                    return None, None, None, None
                debugplotter.clear_plot()
                
            elif type(debug) is list:
                if  i == debug[0] and j == debug[1]:
                    print (f'debugging {i}, {j}')
                    debugplotter.debug = True
                else:
                    debugplotter.debug = False

                

            #Cleaning the x any y data (removing nan and smoothing the y)
            x,y = sanitize_FCdata(ExtendsForce[i][j])

            hard_contact = np.min(x)
            topog[i][j] = hard_contact
            x = x - hard_contact
    

            #Differentiating the data and smoothing
            dy = savgol_filter(y, deriv=1, **calculate_savgol_params(x))
            # units in Newtons per meter (I think)
    
            #Ensuring that the script identifies all spikes in gradient,
            #either positive or negative
            dy_abs = abs(dy)
            
            #Idenitifying the peaks in the (abs) first derivative
            peaks, ___ = scipy.signal.find_peaks(dy_abs,
                                                 height=0.2, # n/m
                                                 distance=len(x)/50)

            # if np.argmin(dy) == 0: #IG: Not sure what this is doing
            #     peaks = np.insert(peaks,0,np.argmin(dy))
                
            region_type = np.zeros_like(y)

            if len(peaks) == 0:
                dropin_loc[i][j] = 0

            else:
                if peaks[0] > 5:
                    # Make sure there is a 'peak' at the constant compliance region.
                    peaks = np.insert(peaks,0,0)
                
            
                for k in range(len(peaks)-1):
                    #Selecting the derivative values between two peaks
                    # peak_diff_range = x[peaks[k]:peaks[k+1]]
                    #peak_range = y[peaks[k]:peaks[k+1]]
                    #Calculating how much of the values are positive
                    derivative_percent = np.average(dy[peaks[k]:peaks[k+1]])
                    force_average = np.average(y[peaks[k]:peaks[k+1]])
                    
                    #Derivative_percent function looks at the derivative between adjacent peaks 
                    #and determines whether the slope is positive or not
                    #By summing these increments over a region, we determine if this has an oil or bubble signature
                    
                    #Different cases considered for deciding if oil or gas and
                    #assigning it a generic placeholder
  
                    
                    if derivative_percent < 0.6 and force_average > -0.5e-7: # arb - FIXME
                        region_type[peaks[k]:peaks[k+1]] = Gas
                    
                    else:
                        region_type[peaks[k]:peaks[k+1]] = Oil

    
                dropin_loc[i][j] = x[peaks[-1]]
            #Using the placeholders to find what height value corresponds to the 
            #top of the bubble and gas
            if sum(region_type == Gas) != 0:
                bubble_loc = np.where(region_type == Gas)
                if x[bubble_loc[0][-1]] > 9e-6: #Just avoiding the initial bit of the force curve
                    x[bubble_loc[0][-1]] = 0
                bubble_height[i][j] = x[bubble_loc[0][-1]]
    
            if sum(region_type == Oil) != 0:
                oil_loc = np.where(region_type == Oil)
                if x[oil_loc[0][-1]] > 9e-6: #Just avoiding the initial bit of the force curve
                    x[oil_loc[0][-1]] = 0
                oil_height[i][j] = x[oil_loc[0][-1]]
                
            debugplotter.plot( curves=[[x,y]], labels=['Fresh'], clear=False, ax=1, color='k', ax_xlabel='Separation (m)')
            debugplotter.plot( curves=[[x, dy_abs]], labels=['abs dy'], clear=False, ax=2, color='r', ax_ylabel='Region Type')
            debugplotter.scatter([[x, region_type], [x[peaks], np.zeros_like(peaks)]], labels=['region type', 'peaks'], ax=2)

    
                    
    
                
    return(dropin_loc,bubble_height, oil_height, topog)

def flatten_planefit (topog_map, verbose=False):
    # logic from https://stackoverflow.com/questions/35005386/fitting-a-plane-to-a-2d-array

    
    m = topog_map.shape[0]
    n = topog_map.shape[1]
    s = m*n

    X1, X2 = np.mgrid[:m, :n]

    X = np.hstack((np.reshape(X1, (s, 1)) , np.reshape(X2, (s, 1)) ) )
    X = np.hstack((np.ones((s, 1)) , X ))

    YY = np.reshape(topog_map, (m*n, 1))

    theta = np.dot(np.dot( np.linalg.pinv(np.dot(X.transpose(), X)), X.transpose()), YY)

    plane = np.reshape(np.dot(X, theta), (m, n));

    flat_topog = topog_map - plane
    
    if verbose:
        return flat_topog, plane
    else:
        return flat_topog

def sanitize_FCdata (FCdata):
    """
    removes NaNs and smooths 
    """
    x, y  = FCdata
    x = x[~np.isnan(y)]
    y = y[~np.isnan(y)]
    
    y = savgol_filter(y, **calculate_savgol_params(x))
    
    return x, y
            

def calculate_savgol_params(x_axis, polyorder=2):
    spacing = x_axis[1] - x_axis[0] # in meters
    window_length =  int(len(x_axis)/50)
    
    # ensure window length is at least 5nm
    if window_length * spacing < 5*1e-9:
        window_length = int(5*1e-9/spacing)
        
    # ensure window length is odd
    if window_length%2 == 0:
        window_length += 1
        
    #print (window_length, polyorder, spacing)
        
    return {'window_length':window_length,
            'polyorder':polyorder,
            'delta':spacing}

def heatmap2d(arr, file_name, metadict, newpath='./', postnomial='', save_heatmap=True):
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
    
    file_name += '_' + postnomial

    
    #Plotting the heatmap
    plt.figure()
    plt.suptitle(postnomial)
    plt.imshow(arr, cmap='magma',extent=[0,image_size,0,image_size])
    cbar = plt.colorbar()
    cbar.set_label('Thickness ($\mu$m)', rotation = 270, labelpad = 20)
    plt.xlabel('x ($\mu$m)')
    plt.ylabel('y ($\mu$m)')
    plt.xticks(np.arange(0,image_size+1,image_size/4))
    plt.yticks(np.arange(0,image_size+1,image_size/4))
    if save_heatmap == True:
        save_name = file_name+'heatmap'+'.png'
        plt.savefig(newpath_map + '/' + save_name)
    plt.rcParams['figure.dpi'] = 1000
    plt.show()
    
def forcemapplot(data, coords, file_name, dropin_loc, bubble_height, oil_height, topog, newpath='./', postnomial='', save_forcemap=True):
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
    #jump_in = dropin_loc[coord_x,coord_y]/1e-6
    bubble_h = bubble_height[coord_x,coord_y]/1e-6
    oil_h  = oil_height[coord_x,coord_y]/1e-6
    topog = topog[coord_x,coord_y]/1e-6
    

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
    #plt.xlim((0,1.5*np.max(oil_height)*1e6))
    #plt.axvline(jump_in, c='tab:red', label = 'Initial Jump-in')
    plt.axvline(bubble_h, c='tab:green', label = 'Bubble Height')
    plt.axvline(oil_h, c='tab:orange', label = 'Oil Height')
    plt.legend()
    
    #Saving out the force curve
    if save_forcemap == True:
        save_name = file_name + coords + 'forcecurve'+'.png'
        plt.savefig(newpath_forcecurve + '/' + save_name)
    plt.show()


def side_profile(heights, row, metadict, points_per_line, file_name, newpath='./', postnomial='', horizontal=True, save_sideprofile=True):
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
    plt.plot(x,oil_y,'x-',c='tab:blue', label = 'Oil Height')
    plt.plot(x,bubble_y,'x-', c = 'tab:red', label = 'Bubble Height')
    plt.rcParams['figure.dpi'] = 500
    plt.xlabel('x ($\mu$m)')
    plt.ylabel('Height ($\mu$m)')
    plt.legend()
    
    if save_sideprofile == True:
        save_name = file_name + 'side_profile'+str(row)+'.png'
        plt.savefig(newpath_sideprofile + '/' + save_name)
    plt.show()
    
def is_MAC(file_name = '',date_taken = '',mac = True, initiator = False):
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
        newpath = r'L:\ljam8326 Asylum Research AFM\Infused Teflon Wrinkle Samples in Air\230721 Samples'
    else:
        newpath = r'C:\Users\Seamu\OneDrive\University\USYD (2021-)\Honours\AFM Data Processing/' 
        
    os.chdir(newpath)
    if initiator == False:
        newpath = newpath + date_taken + '/' +file_name
    return newpath
