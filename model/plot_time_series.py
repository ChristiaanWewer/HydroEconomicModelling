import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

# load dataset
forcing = np.genfromtxt(r'INPUT_DATA/IMD_1975_2013.txt',  dtype=float, autostrip=True)
save_loc = r'{}/output/input.png'.format(os.path.dirname(os.getcwd()))
mpl.use('Qt5Agg')

# get variables
forcing2013 = forcing[13880:14245,]
Ep = forcing2013[:,2]
P = forcing2013[:,1]
Q = forcing2013[:,0]

P_bar = []
for i, p in enumerate(P):
    P_bar.append(None)
    P_bar.append(0)
    P_bar.append(p)

P_bars = np.array(P_bar)

x = np.arange(len(P))
x_bar = np.repeat(x,3)

# plot precipitation and evaporation
fig, ax1 = plt.subplots(figsize=(15,7.5))

# make different axes
ax11 = ax1.twinx()
ax11.invert_yaxis()
ax12 = ax1.twinx()
ax11.spines.right.set_position(("axes", 1.05))

# plot
l3 = ax12.plot(x, Q, label='Measured Discharge (bottom)')
l4 = ax1.plot(x, Ep, label='Potential Evaporation', color='darkgreen')
l5 = ax11.plot(x_bar, P_bars, label='Measured Precipitation (top)')
ax1.set_ylim(-0.1, 7)
ax11.set_ylim(250, 0)
ax12.set_ylim(0,150)

# labels
ax1.set_ylabel('Evaporation (mm/d)')
ax11.set_ylabel('Precipitation (mm/d)')
ax12.set_ylabel('Discharge (m3/d)')
ax1.set_xlabel('Day of year (d)')

# legend
lt1 = l3 + l4 + l5
ax1.legend(lt1, [l.get_label() for l in lt1], loc='lower right')
ax1.set_title('Precipitation, Evaporation and Discharge')
fig.tight_layout()
fig.savefig(save_loc)
