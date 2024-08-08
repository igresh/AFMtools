# import nanoscope as ns
import numpy as np
import os
import csv
import copy
from scipy.signal import savgol_filter

from nanoscope import files
from nanoscope.constants import FORCE, METRIC, VOLTS, PLT_kwargs, RAW

import sys
sys.path.append('/Users/isaac/Documents/GitHub/AFMtools')
import ForceCurveFuncs
import ImageFuncs

def PeakforceImport(Filename, output_dir='./Output'):
    """
    Uses the nanoscope module (provided by bruker) to open the peakforce data files.

    Creates a new directory in the output_dir folder using the name of the peakforce
    file. The directory contains the following files:
        -   image.npy (numpy binary) the raw image from the 'image channel'.
            A NxN array containing the topography (height) of the substrate, where N
            is the scan size
            
        -   qnmcurves.npy (numpy binary) the force curves for the image.
            A NxNxYx2 array where N is the scan size and Y is the number of datapoints
            in each force curve.

        -   values.csv (csv file) Relevant imaging parameters.
    """
    arr = []

    if not os.path.exists(output_dir):
        print ('Creating output directory...')
        os.mkdir(output_dir)
            
    name = '.'.join(os.path.basename(Filename).split('.')[:-1])
    print (name)

    if not os.path.exists(f'{output_dir}/{name}'):
        os.mkdir(f'{output_dir}/{name}')
    
    with files.PeakforceCaptureFile(Filename) as file:
        image_channel = file.image_channel
        fv_image, ax_properties = image_channel.create_image(METRIC)
    
        spl = image_channel.samples_per_line
        lines = image_channel.number_of_lines
    
        fc_channel = file.force_curves_channel
        fv_pixels = fc_channel.number_of_force_curves
    
        for pix_idx in range(fv_pixels):
            fz_plot_bl, _ = fc_channel.create_force_z_plot(pix_idx, FORCE)
            fs_plot_bl = fc_channel.compute_separation(fz_plot_bl, FORCE)
    
            arr.append([fs_plot_bl.trace.x,
                        fs_plot_bl.trace.y,
                        fs_plot_bl.retrace.x,
                        fs_plot_bl.retrace.y])
    
    
        values_of_interest = {'image scan size':image_channel.scan_size,
                              'image scan size unit':image_channel.scan_size_unit,
                              'image line number':image_channel.number_of_lines,
                              'samples per line':image_channel.samples_per_line,
                              'spring constant':image_channel.spring_constant,
                              'optical sensitivity?':image_channel.z_scale_in_sw_units}
    
    arr = np.array(arr, dtype=np.float16 )
    
    np.save(f'{output_dir}/{name}/qnmcurves', arr)
    # np.save(f'{output_dir}/{name}/image', fv_image)
    save_array(data=fv_image, name='image', directory=f'{output_dir}/{name}')

    with open(f'{output_dir}/{name}/values.csv', 'w', newline="", encoding='utf-8') as f:  
        writer = csv.DictWriter(f, fieldnames=values_of_interest.keys())
        writer.writeheader()
        writer.writerow(values_of_interest)



def processForceMap(direc):
    with open(f'{direc}/values.csv', "r", encoding='utf-8') as infile:
        reader = csv.DictReader(infile)

        for row in reader:
            val_dict = row
        
    scan_size = float(val_dict['image scan size'])
    scan_unit = val_dict['image scan size unit']
    
    arr = np.load(f'{direc}/qnmcurves.npy')

    ExtendsForce = copy.deepcopy(arr[:,0:2])
    RetractsForce = copy.deepcopy(arr[:,2:4])

    baseline_av = np.average(ExtendsForce[:,1,:50], axis=1)
    points_per_line = int(np.sqrt(len(arr)))

    ExtendsForce[:,1] = (ExtendsForce[:,1].T - baseline_av.T).T
    RetractsForce[:,1] = (RetractsForce[:,1].T - baseline_av.T).T

    ExtendsForce[:,1] = ExtendsForce[:,1,::-1]                          # Reverse
    ExtendsForce[:,0] = ExtendsForce[:,0,::-1]                          # Reverse
    # RetractsForce[:,1] = RetractsForce[:,1,::-1]    # reverse 
    # RetractsForce[:,0] = RetractsForce[:,0,::-1]                      # Reverse

    sav_params = {'window_length':5, 'polyorder':1}
    ExtendsForce[:,1] = savgol_filter(ExtendsForce[:,1], **sav_params)
    RetractsForce[:,1] = savgol_filter(RetractsForce[:,1], **sav_params)
    
    
    jump_in = []
    pull_off = []
    wadh_in = []
    wadh_off = []
    rep_on  = []
    rep_off = []

    for idx, [EF, RF] in enumerate(zip(ExtendsForce, RetractsForce)):
        # Calculate jump in
        mask = EF[1]<np.min(EF[1,:50])*0.5
        mask[70:] = False
        if np.sum(mask)==0:
            jump_in.append(0)
        else:
            jump_in.append(np.quantile(EF[0][mask],q=0.98))

        # Calculate work of attraction
        mask = EF[0] < jump_in[-1]+10
        newEF = np.copy(EF)
        newEF[1][newEF[1]>0]=0

        if np.sum(mask)==0:
            wadh_in.append(0)
        else:
            wadh = np.trapz(newEF[1], newEF[0])
            if not wadh == np.inf:
                wadh_in.append(wadh)
            else:
                print ('inf')
                wadh_in.append(0)

        
        # Calculate the location of repulsive onset
        mask = np.logical_and(EF[0] < jump_in[-1], EF[1]>0)

        if np.sum(mask)==0:
            rep_on.append(0)
        else:
            idx_at_int = np.argwhere(mask)[-1][0]
            intercept = np.interp(x=0, xp=EF[1,idx_at_int:idx_at_int+2], fp=EF[0,idx_at_int:idx_at_int+2])
            rep_on.append(intercept)


        # calculate the point of pull-off    
        mask = RF[1]<np.min(RF[1,:50])*0.5
        mask[70:] = False
        if np.sum(mask)==0:
            pull_off.append(0)
        else:
            pull_off.append(np.quantile(RF[0][mask],q=0.98))


        # Calculate work of adhesion
        mask = RF[0] < pull_off[-1]+10
        newRF = np.copy(RF)
        newRF[1][newRF[1]>0]=0

        if np.sum(mask)==0:
            wadh_off.append(0)
        else:
            wadh = np.trapz(newRF[1], newRF[0])
            if not wadh == np.inf:
                wadh_off.append(wadh)
            else:
                print ('inf')
                wadh_off.append(0)


        # Calculate the location of repulsive offset
        mask = np.logical_and(RF[0] < pull_off[-1], RF[1]>0)
        if np.sum(mask)==0:
            rep_off.append(0)
        else:
            idx_at_int = np.argwhere(mask)[-1][0]
            intercept = np.interp(x=0, xp=EF[1,idx_at_int:idx_at_int+2], fp=EF[0,idx_at_int:idx_at_int+2])
            rep_off.append(intercept)


    jump_in  =  np.array(jump_in, dtype=np.float64)
    pull_off =  np.array(pull_off, dtype=np.float64)
    wadh_in  = - 1e-9 * np.array(wadh_in, dtype=np.float64) # report value in nJ
    wadh_off = - 1e-9 * np.array(wadh_off, dtype=np.float64) # report value in nJ
    rep_on   =  np.array(rep_on, dtype=np.float64)
    rep_off  =  np.array(rep_off, dtype=np.float64)

    
    ExtendsForce = np.reshape(ExtendsForce,(points_per_line,points_per_line,2,-1))
    RetractsForce = np.reshape(RetractsForce,(points_per_line,points_per_line,2,-1))


    ExtendsAdh = -np.min(ExtendsForce, axis=3)[:,:,1]
    RetractsAdh = -np.min(RetractsForce, axis=3)[:,:,1]
    wadh_in_arr = np.squeeze(np.reshape(wadh_in,(points_per_line,points_per_line,-1)))
    wadh_off_arr = np.squeeze(np.reshape(wadh_off,(points_per_line,points_per_line,-1)))
    jump_in_arr = np.squeeze(np.reshape(jump_in,(points_per_line,points_per_line,-1)))
    pull_off_arr = np.squeeze(np.reshape(pull_off,(points_per_line,points_per_line,-1)))
    rep_on_arr = np.squeeze(np.reshape(rep_on,(points_per_line,points_per_line,-1)))
    rep_off_arr = np.squeeze(np.reshape(rep_off,(points_per_line,points_per_line,-1)))
    
    save_array(data=ExtendsAdh, name='extends_adhesion', directory=direc)
    save_array(data=RetractsAdh, name='retracts_adhesion', directory=direc)
    save_array(data=wadh_in_arr, name='work_of_attraction', directory=direc)
    save_array(data=wadh_off_arr, name='work_of_adhesion', directory=direc)
    save_array(data=jump_in_arr, name='jump_in', directory=direc)
    save_array(data=pull_off_arr, name='jump_off', directory=direc)
    save_array(data=rep_on_arr, name='net_repulsion_in', directory=direc)
    save_array(data=rep_off_arr, name='net_repulsion_off', directory=direc)
    
    save_array(data=ExtendsForce, name='extend_force_curves', directory=direc, savecsv=False)
    save_array(data=RetractsForce, name='retract_force_curves', directory=direc, savecsv=False)

    

def save_array(data, name, directory,savecsv=True):
    np.save(f'{directory}/{name}.npy', data)
    if savecsv:
        np.savetxt(f'{directory}/{name}.csv', data, delimiter=',')

def get_bounds(A, B):
    bmin = np.min([np.mean(A) - np.std(A), np.mean(B) - np.std(B)])
    bmax = np.max([np.mean(A) + np.std(A), np.mean(B) + np.std(B)])
    return bmin, bmax