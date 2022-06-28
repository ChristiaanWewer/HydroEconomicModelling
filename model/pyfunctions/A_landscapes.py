import numpy       as np
import matplotlib as mpl
import matplotlib.pyplot as plt


#cd C:\Users\chris\Documents\mastervakken\CIE4400 Hydroeconomic Modelling\group project\model\python_solutions\solution


DEM=np.genfromtxt(r'INPUT_DATA/demmuthk.asc',  dtype=float, autostrip=True)
slope=np.genfromtxt(r'INPUT_DATA/slopemuthk.asc',  dtype=float, autostrip=True)
hand=np.genfromtxt(r'INPUT_DATA/handmuthk.asc',  dtype=float, autostrip=True)
basin=np.genfromtxt(r'INPUT_DATA/basin.asc',  dtype=float, autostrip=True)
LULC=np.genfromtxt(r'INPUT_DATA/lulc_muthk.asc')

#plot DEM
plt.figure(1)
DEM[DEM==-9999]=np.nan
plt.imshow(DEM, cmap='jet')
plt.title('DEM')
plt.colorbar()
plt.show()

 
#plot HAND
plt.figure(2)
hand[hand==-9999]=np.nan
plt.imshow(hand, cmap='jet')
plt.title('HAND')
plt.colorbar()
plt.show()


#plot slope
plt.figure(3)
slope[slope==-9999]=np.nan
plt.imshow(slope, cmap='jet')
plt.title('slope')
plt.colorbar()
plt.show()


#plot LULC
plt.figure(4)
LULC[LULC==-9999] = np.nan
plt.imshow(LULC, cmap='jet')
plt.title('LULC')
plt.colorbar()
plt.show()


#make landscape classification
hillslope = np.array(slope) >4
plateau = (np.array(hand) > 1) & (np.array(slope) <= 4)
wetland = (np.array(hand) <= 1) & (np.array(slope) <= 4)
basin = np.array(basin)>0

hillslope_per = float(np.sum(hillslope))/float(np.sum(basin))
wetland_per = float(np.sum(wetland))/float(np.sum(basin))
plateau_per = float(np.sum(plateau))/float(np.sum(basin))



landscapes=np.zeros((482,573))
landscapes[plateau]=1
landscapes[hillslope]=2
landscapes[wetland]=3

#plot landscapes
cmap = mpl.colors.ListedColormap(['white', 'red', 'green', 'blue'])
bounds=[0,0.75,1.5,2.25,3]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

plt.figure(5)
plt.imshow(landscapes, cmap=cmap, norm=norm)
plt.title('landscape')
plt.colorbar()
plt.show()





