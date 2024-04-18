#%%
file_name = "SW_S3UBA03.ARDF"
nThresh = 1000
percent = 20

#%%
raw, defl, metadict = data_load_in(file_name)

#%%
ExtendsForce, points_per_line = data_convert(raw, defl, metadict)

#%%
dropin_loc, bubble_height, oil_height = data_process(ExtendsForce, points_per_line,nThresh,percent)

heatmap2d(dropin_loc,10,f_name = 'd'+file_name, save = False)
heatmap2d(bubble_height,10,f_name = 'b'+file_name, save = False)
heatmap2d(oil_height,10,f_name = 'o'+file_name, save = False)
#%%
forcemapplot(ExtendsForce[70][70],(70,70),f_name = file_name, save = True)


#%%
        
x,y = ExtendsForce[75][40]
x = x[~np.isnan(y)]
y = y[~np.isnan(y)]

plt.figure()
plt.plot(x,y)

y = savgol_filter(y, 51, 2)

dy = np.diff(y)
dy = savgol_filter(dy, 51, 2)



dx = x[range(len(dy))]

d2y = np.diff(dy)
d2y = savgol_filter(d2y,81,2)
x1 = range(len(dy))
x2 = range(len(d2y))

dx = x[range(len(dy))]

peaks = scipy.signal.find_peaks(dy,1e-10, distance = 50, prominence = 1e-11)
peaks = peaks[0]
if np.argmin(dy) == 0:
    peaks = np.insert(peaks,0,np.argmin(dy))
region_type = np.zeros(len(y))

plt.figure()
plt.plot(dx,dy)
plt.axvline(dx[peaks[0]], c='tab:red')
plt.axvline(dx[peaks[1]], c='tab:green')
plt.axvline(dx[peaks[-1]], c='tab:orange')

if len(peaks) == 0:
    print('Water')
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


a = np.where(region_type == 2)

print(x[a[0][-1]])


print(x[peaks[-1]])
#%%
plt.figure()
plt.plot(x[range(len(d2y))],d2y)
plt.ylim((-1e-11,1e-11))


#%%
for f in range(70,79):
    for l in range(70,79):
        print((f,l))
        
        
#%%
cross_bubble = np.zeros(points_per_line)
cross_oil = np.zeros(points_per_line)
row_val = 73
x_range = np.linspace(0,10,points_per_line)


for g in range(points_per_line):
    cross_bubble[g] = bubble_height[row_val][g]/1e-6
    cross_oil[g] = oil_height[row_val][g]/1e-6

plt.figure()
plt.scatter(x_range,cross_bubble,marker = 'x',label = 'Bubble Height')