# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 10:25:11 2023

@author: Leo James
"""
import sys
sys.path.append(r'C:\Users\leoja\OneDrive\Documents\GitHub\AFMtools')

import numpy as np
import matplotlib.pyplot as plt
import load_ardf  
from InterfaceCore import data_load_in, data_convert, data_process, heatmap2d, forcemapplot, side_profile
import ForceCurveVisualisation
import ForceCurveFuncs

#%%
file_name = r"L:\ljam8326 Asylum Research AFM\Infused Teflon Wrinkle Samples in Air\230721 Samples\TWPS5cSt00"
metadict = load_ardf.metadict_output(file_name+'.ARDF')
date_taken = metadict["LastSaveForce"][-7:-1]

save_forcemap = True
save_heatmap = True
save_sideprofile = False

x_pos, y_pos = (8,3)

#%%
raw, defl, metadict = data_load_in(file_name)
ExtendsForce, points_per_line = data_convert(raw, defl, metadict, zero_constant_compliance=False)


#%%
fig, ax = plt.subplots()

reshaped =  np.reshape(ExtendsForce, (-1, 2, 1000))
x, y = reshaped[1]
ax.plot(x,y, color='k')
        
#%%
dropin_loc, bubble_height, oil_height, topog = data_process(ExtendsForce, points_per_line)

#%%
# bubble_height[43,15] = 
#oil_height[42,27] = 0
heatmap2d(bubble_height,file_name, metadict, postnomial='Bubble Thickness')
heatmap2d(oil_height,file_name, metadict, postnomial='Lubricant Thickness')
heatmap2d(topog,file_name, metadict, postnomial='Topography')


#%%
x_pos,y_pos = [43,15]
forcemapplot(ExtendsForce[x_pos][y_pos],(x_pos,y_pos), file_name, dropin_loc, bubble_height, oil_height, topog)

#%%
side_profile([oil_height,bubble_height],6, metadict, points_per_line, file_name)

#%%
raw, defl, metadict = data_load_in(file_name)
FCD = ForceCurveFuncs.process_zpos_vs_defl(raw, defl, metadict, failed_curve_handling = 'retain',
                                      zero_at_constant_compliance=False,   )
 