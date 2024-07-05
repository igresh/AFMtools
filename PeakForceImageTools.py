import numpy as np
import csv
import sys
sys.path.append('C:\\Users\\Isaac\\Documents\\AFMtools')
sys.path.append('/Users/isaac/Documents/GitHub/AFMtools/')
import ImageFuncs
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.patches as mpatches


def construct_subplots():
    """
    Makes basic subplots for a square image with additional 
    axis for a color bar.

    returns:
        fig (matploblib.fig)
        ax (matplotlib.axis): Axis to plot the image on
        cax (matplotlib.axis): Axis to plot the colourbar in
    """
    fig = plt.figure(figsize=(3.6,3))
    gs = GridSpec(1,2, figure=fig, width_ratios=[1,0.05], wspace=0.05)
    fig.subplots_adjust(left=0.01, bottom=0.03, right=0.8, top=0.93)
    
    ax = fig.add_subplot(gs[0])
    cax = fig.add_subplot(gs[1])

    return fig, ax, cax 


def plot_single_image(ax, cax, data, values, bounds=None, rel_bounds=False, top_right_text='', top_left_text='', cbar_label='Height, nm', ar=1, print_median_value=True, scale_bar_size=None):
    if scale_bar_size:
        sbsize = np.round(scale_bar_size)
    else:
        sbsize = np.round(values["scan size"]/3)

    scalebar_settings = {'size':sbsize,
                         'label':f'{str(sbsize)[:-2]} {values["scan unit"]}',
                         'loc':'lower right', 
                         'pad':0.3,
                         'color':'white',
                         'frameon':False,
                         'size_vertical':values["scan size"]/50}

    im_settings = {'interpolation':'bicubic',
                   'extent':[0, values["scan size"], 0, values["scan size"]], 
                   'aspect':ar}

    av_height = np.median(data)
    if bounds:
        if rel_bounds:
            im_settings['vmin']  = bounds[0] + av_height
            im_settings['vmax']  = bounds[1] + av_height
        else:
            im_settings['vmin']  = bounds[0]
            im_settings['vmax']  = bounds[1]


    im=ax.imshow(data, **im_settings)
    scalebar = AnchoredSizeBar(ax.transData, **scalebar_settings)
    ax.add_artist(scalebar)
    ax.set_axis_off()
    
    ax.text(s=top_left_text,x=0,y=1.01, transform=ax.transAxes, va='bottom', size=7)
    ax.text(s=top_right_text,x=1,y=1.01, transform=ax.transAxes, va='bottom', ha='right', size=7)

    if print_median_value:
        ax.text(s=f"median value: {np.round(av_height,2)}",x=1,y=0.99, transform=ax.transAxes, va='top', ha='right', size=7,
               color='white')

    cax = plt.colorbar(mappable=im, cax=cax)
    cax.set_label(cbar_label, fontsize=8)
    cax.ax.tick_params(labelsize=8)

    

def plot_images_summary(imagename, dir='Output'):
    images, values = open_peakforce_images(imagename, dir=dir)
    ExtendsForce = np.load(f'{dir}/{imagename}/extend_force_curves.npy')
    RetractsForce = np.load(f'{dir}/{imagename}/retract_force_curves.npy')

    savefig_settings = {'transparent':True, 'dpi':300}


    # Set parameters
    x,y = 50,50
    scans_per_line = 256
    scansize = values['scan size']
    scale = values['scan size']/scans_per_line

    xpic, ypic = x*scale, values['scan size']-scale*y

    topo_bounds = [-2,2]
    netrep_bounds = [-1,1]

    width = scansize*0.1


    # Sort out figures and subplots
    fig = plt.figure(figsize=(11,7))
    fig.subplots_adjust(left=0.01, bottom=0.08, right=0.94, top=0.96)

    spacing = 0.3
    gs = GridSpec(3, 11, width_ratios=[1, 0.05, spacing, 1, 0.05, spacing, 1, 0.05, spacing, 1, 0.05], height_ratios=[1,1,1], wspace=0.05, hspace=0.2)

    ax1   = fig.add_subplot(gs[0,0])
    ax1cb = fig.add_subplot(gs[0,1])

    ax2   = fig.add_subplot(gs[0,3])
    ax2cb = fig.add_subplot(gs[0,4])

    ax3   = fig.add_subplot(gs[0,6])
    ax3cb = fig.add_subplot(gs[0,7])

    ax4   = fig.add_subplot(gs[0,9])
    ax4cb = fig.add_subplot(gs[0,10])

    ax5   = fig.add_subplot(gs[1,0])
    ax5cb = fig.add_subplot(gs[1,1])

    ax6   = fig.add_subplot(gs[1,3])
    ax6cb = fig.add_subplot(gs[1,4])

    ax7   = fig.add_subplot(gs[2,0])
    ax7cb = fig.add_subplot(gs[2,1])

    ax8   = fig.add_subplot(gs[2,3])
    ax8cb = fig.add_subplot(gs[2,4])

    force_curve_axis_ext = fig.add_subplot(gs[1,6:11])
    force_curve_axis_ret = fig.add_subplot(gs[2,6:11])

    force_curve_axis_ext.yaxis.tick_right()
    force_curve_axis_ret.yaxis.tick_right()
    force_curve_axis_ext.yaxis.set_label_position("right")
    force_curve_axis_ret.yaxis.set_label_position("right")
    force_curve_axis_ext.text(0.98,0.98,s=f'Approach\npeakforce: {str(np.max(ExtendsForce[x,y, 1]))[:5]} nN\nDistance: {str(np.max(ExtendsForce[x,y, 0]))[:5]} nm', ha='right', va='top', transform=force_curve_axis_ext.transAxes)
    force_curve_axis_ret.text(0.98,0.98,s='Retract', ha='right', va='top', transform=force_curve_axis_ret.transAxes)

    ## Force curve
    force_curve_axis_ext.scatter(ExtendsForce[x,y, 0], ExtendsForce[x,y, 1], color='k', s=5)
    force_curve_axis_ret.scatter(RetractsForce[x,y, 0], RetractsForce[x,y, 1], color='k', s=5)
    force_curve_axis_ext.set(ylabel='Force, nN',
                             xlim=(-5,60), ybound=(-5,5))
    force_curve_axis_ret.set(ylabel='Force, nN', xlabel='tip-substrate separation, nm',
                             xlim=(-5,60), ybound=(-5,5))
    force_curve_axis_ret.axvspan(-1,0, color='k', alpha=0.3)
    force_curve_axis_ext.axvspan(-1,0, color='k', alpha=0.3)


    # Topography
    image = ImageFuncs.flatten(images['image'], retain_magnitude=True)
    plot_single_image(ax1, ax1cb, image, values, rel_bounds=True, bounds=topo_bounds,
                      top_right_text='', top_left_text='Topography')
    ax1.scatter(xpic,ypic, edgecolor='r')# Figure out how to do offets
    artist = mpatches.Rectangle((0, 0), width, width, ec="none", color='gray')
    ax1.add_artist(artist)


    # Adhesion
    image = ImageFuncs.flatten(images['Retracts Adh'], retain_magnitude=True)
    plot_single_image(ax2, ax2cb, image, values,
                      top_right_text='', top_left_text='Retract max adhesion force',
                      cbar_label='Force, nN')
    force_curve_axis_ret.axhline(y=-image[x,y], color='r')
    artist = mpatches.Rectangle((0, 0), width, width, ec="none", color='r')
    ax2.add_artist(artist)


    # Jump in
    image = ImageFuncs.flatten(images['jump in'], retain_magnitude=True)
    plot_single_image(ax3, ax3cb, image, values, rel_bounds=True, bounds=topo_bounds,
                      top_right_text='', top_left_text='Jump-in')
    force_curve_axis_ext.axvline(x=image[x,y], color='xkcd:light blue')
    artist = mpatches.Rectangle((0, 0), width, width, ec="none", color='xkcd:light blue')
    ax3.add_artist(artist)

    # Jump off
    image = ImageFuncs.flatten(images['pull off'], retain_magnitude=True)
    plot_single_image(ax4, ax4cb, image, values, rel_bounds=True, bounds=topo_bounds,
                      top_right_text='', top_left_text='Jump-off')
    force_curve_axis_ret.axvline(x=image[x,y], color='xkcd:blue')
    artist = mpatches.Rectangle((0, 0), width, width, ec="none", color='xkcd:blue')
    ax4.add_artist(artist)

    # Work of attraction
    image = ImageFuncs.flatten(images['wadh in'], retain_magnitude=True)
    plot_single_image(ax5, ax5cb, image, values, top_right_text='', top_left_text='Work of attraction', cbar_label='Work, nJ')

    # Work of adhesion
    image = ImageFuncs.flatten(images['wadh off'], retain_magnitude=True)
    plot_single_image(ax6, ax6cb, image, values, top_right_text='', top_left_text='Work of adhesion', cbar_label='Work, nJ')


    # Net repulsion in
    image = ImageFuncs.flatten(images['net rep in'], retain_magnitude=True)
    plot_single_image(ax7, ax7cb, image, values, rel_bounds=True, bounds=netrep_bounds,
                      top_right_text='', top_left_text='Start of net repulsion (approach)')
    force_curve_axis_ext.axvline(x=image[x,y], color='xkcd:light orange')
    artist = mpatches.Rectangle((0, 0), width, width, ec="none", color='xkcd:light orange')
    ax7.add_artist(artist)

    image = ImageFuncs.flatten(images['net rep off'], retain_magnitude=True)
    plot_single_image(ax8, ax8cb, image, values, rel_bounds=True, bounds=netrep_bounds,
                      top_right_text='', top_left_text='Start of net repulsion (retract)')
    force_curve_axis_ret.axvline(x=image[x,y], color='xkcd:orange')
    artist = mpatches.Rectangle((0, 0), width, width, ec="none", color='xkcd:orange')
    ax8.add_artist(artist)

    fig.text(0.02,0.05, s=f'scan size: {values["scan size"]} {values["scan unit"]}, samples per line: {values["samples per line"]}, spring constant: {values["spring constant"]} nN/nm', ha='left', va='bottom')
    fig.text(0.019,0.05, s=f'{imagename}', ha='left', va='top')

    fig.savefig(f'{dir}/{imagename}/image_summary.png', **savefig_settings)
    fig.savefig(f'{dir}/{imagename}/image_summary.pdf', **savefig_settings)

    return fig



def open_peakforce_images(imagename, dir='Output'):
    """
    returns:
        images (dict): Dictionary of all images
        values (dict): Dictionary of relevant imaging paramters
    """
    path = f'{dir}/{imagename}'
    with open(f'{path}/values.csv', "r", encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
    
        for row in reader:
            val_dict = row

    values = {'scan size':float(val_dict['image scan size']),
              'scan unit':val_dict['image scan size unit'],
              'spring constant':val_dict['spring constant'],
              'samples per line':val_dict['samples per line']}
    
    images = {
        'image'             :  np.load(f'{path}/image.npy').astype('float32'),
        'Extends Adh'       :  np.load(f'{path}/extends_adhesion.npy').astype('float32'),
        'Retracts Adh'      :  np.load(f'{path}/retracts_adhesion.npy').astype('float32'),
        'wadh off'          :  np.load(f'{path}/work_of_adhesion.npy').astype('float32'),
        'wadh in'           :  np.load(f'{path}/work_of_attraction.npy').astype('float32'),
        'jump in'           :  np.load(f'{path}/jump_in.npy').astype('float32'),
        'pull off'          :  np.load(f'{path}/jump_off.npy').astype('float32'),
        'net rep in'        :  np.load(f'{path}/net_repulsion_in.npy').astype('float32'),
        'net rep off'       :  np.load(f'{path}/net_repulsion_off.npy').astype('float32'),
        'flat image'        :  ImageFuncs.flatten(np.load(f'{path}/image.npy').astype('float32'), retain_magnitude=True),
        'flat Extends Adh'  :  ImageFuncs.flatten(np.load(f'{path}/extends_adhesion.npy').astype('float32'), retain_magnitude=True),
        'flat Retracts Adh' :  ImageFuncs.flatten(np.load(f'{path}/retracts_adhesion.npy').astype('float32'), retain_magnitude=True),
        'flat wadh off'     :  ImageFuncs.flatten(np.load(f'{path}/work_of_adhesion.npy').astype('float32'), retain_magnitude=True),
        'flat wadh in'      :  ImageFuncs.flatten(np.load(f'{path}/work_of_attraction.npy').astype('float32'), retain_magnitude=True),
        'flat jump in'      :  ImageFuncs.flatten(np.load(f'{path}/jump_in.npy').astype('float32'), retain_magnitude=True),
        'flat pull off'     :  ImageFuncs.flatten(np.load(f'{path}/jump_off.npy').astype('float32'), retain_magnitude=True),
        'flat net rep in'   :  ImageFuncs.flatten(np.load(f'{path}/net_repulsion_in.npy').astype('float32'), retain_magnitude=True),
        'flat net rep off'  :  ImageFuncs.flatten(np.load(f'{path}/net_repulsion_off.npy').astype('float32'), retain_magnitude=True)}

    return images, values