import sys
sys.path.append('/Users/isaac/Documents/GitHub/AFMtools')

import load_ardf

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# from igor import binarywave

from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib import patheffects

pe = [patheffects.withStroke(linewidth=2,
                             foreground="w")]
    


def open_ibw(fname):
    bw = binarywave.load(fname)['wave']
    im = bw['wData']
    note = bw['note']
    
    
    image_params_dict = {}

    for entry in str(note)[2:].split('\\r'):

        if ': ' in entry:
            tounpack = entry.split(': ')
            if len(tounpack) > 2:
                pname = tounpack[0]
                pvalue = ':'.join(tounpack[1:])
            elif len(tounpack) == 2:
                pname, pvalue = tounpack
        else:
            tounpack = entry.split(':')
            if len(tounpack) > 2:
                pname = tounpack[0]
                pvalue = ':'.join(tounpack[1:])
            elif len(tounpack) == 2:
                pname, pvalue = tounpack
                
        try:
            pvalue = float(pvalue)
        except:
            pvalue = pvalue

        # if len (pvalue) > 1

        image_params_dict[pname] = pvalue
        
    im = np.transpose(im, axes=(2,1,0))
        
    return im, image_params_dict


def calculate_Ra (image):
    """
    image is a N by M ndarray
    
    checked against the roughness calcs in the asylum software - gives the same answer
    """
    
    mean = np.mean(image)
    deviation = np.abs(image - mean)
    return np.sum(deviation) / image.size

def calculate_Rq (image):
    """
    image is a N by M ndarray
    
    checked against the roughness calcs in the asylum software - gives the same answer

    """
    
    return np.std(image)



def calculate_Skew (image):
    """
    Checked against asylum software calcs
    """
    mean = np.mean(image)
    deviation = image - mean
    
    return np.sum(deviation**3) / (image.size*calculate_Rq(image)**3)


def calculate_Kurt (image):
    """
    Checked against asylum software calcs
    """
    mean = np.mean(image)
    deviation = image - mean
    
    # I don't know where the -3 comes from, but its in the asyluym software so I guess I'll put it in here?
    return np.sum(deviation**4) / (image.size*calculate_Rq(image)**4)-3


def add_image_params(image, ax):
    
    Rq = calculate_Rq(image)
    Ra = calculate_Ra(image)
    Skew = calculate_Skew(image)
    Kurt = calculate_Kurt(image)

    
    ax.text(x=0.03, y=0.03, s= f'$R_q=${np.round(Rq,3)}\n$R_a=${np.round(Ra,3)}\nSkewness$=${np.round(Skew,3)}\nKurtosis$=${np.round(Kurt,3)}', va='bottom', transform=ax.transAxes,
           path_effects=pe)
    
    

def flatten (imarr, retain_magnitude=False, order=0):
    """
    Line flattening
    """
    assert order <= 2 , "Orders > 2 not implimented"
    average = np.average(imarr)
    flat_imarr = np.copy(imarr)
    
    for line in flat_imarr:
        if order == 0:
            line -= np.average(line)
        elif order == 1:
            x = np.linspace(1, len(line), len(line))
            m,b = np.polyfit(x, line, 1)
            line -= (x*m+b)
        else:
            x = np.linspace(1, len(line), len(line))
            a,b,c = np.polyfit(x, line, 2)
            line -= (a*x**2+b*x+c)
        
    if retain_magnitude:
        flat_imarr += average

    return flat_imarr


def shift_z(image, shift='mean'):
    if shift == 'mean':
        image -= np.mean(image)
    elif shift == 'median':
        image -= np.median(image)
    elif shift == 'min':
        image -= np.min(image)
    elif shift == 'max':
        image -= np.max(image)
    elif shift[0] =='q':
        quant = int(shift[1:])/100
        image -= np.quantile(image, quant)

    return image


def process_AFM(fname, debug=False):
    images, imdict = open_ibw(fname)
    
    image = images[0] #select height map
    
    if debug:
        imoutname = fname[:-len('.ibw')] + '_height.png'
        plot_AFM(image, imdict, imoutname)
    
    Rq = calculate_Rq(image)*1e9
    Ra = calculate_Ra(image)*1e9
    Skew = calculate_Skew(image)
    Kurt = calculate_Kurt(image)

    return image, {'Rq':Rq, 'Ra':Ra, 'Skewness':Skew, 'Kurtosis':Kurt}


unit_dict = {'nm':1e9,
             'um':1e6,
             'mm':1e3,
             'm':1e0}

def plot_AFM(image, imdict, imoutname=None, cmap='copper', bounds=None, zunit='nm', title=None, ax=None, cax=None):
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(5,4))
    else:
        fig = ax.get_figure()
    fig.suptitle(title)

    if bounds is None:
        bounds = [np.quantile(image,0.01), np.quantile(image,0.99)]
        range = bounds[1]-bounds[0]
        bounds[0] -= range*0.4
        bounds[1] += range*0.4

    scale_adj = unit_dict[zunit] / unit_dict[imdict['z unit']]

    norm = matplotlib.colors.Normalize(vmin=bounds[0], vmax=bounds[1])
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    
    ax.axis("off")
    ax.imshow(image*scale_adj, cmap=cmap, norm=norm)
    scalebar = ScaleBar (imdict['ScanSize']/imdict['ScanPoints'], units=imdict['xy unit'], location='lower right', frameon=False)
    ax.add_artist(scalebar)
    add_image_params(image*scale_adj, ax)

    if cax is not None:
        plt.colorbar(mappable, cax=cax)
    else:
        plt.colorbar(mappable, ax=ax)

    if imoutname is not None:
        fig.savefig(imoutname, dpi=300)
        plt.close(fig)