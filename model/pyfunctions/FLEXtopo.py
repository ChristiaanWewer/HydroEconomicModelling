import numpy       as np
from pyfunctions.Weigfun import Weigfun
from pyfunctions.plateau import plateau
from pyfunctions.hillslope import hillslope
from pyfunctions.wetland import wetland

def FLEXtopo( ParPlateau, ParHillslope_forest, ParHillslope_crop, ParWetland, ParCatchment,forcing, landscapes):


    #parameters and constants
    Ks = ParCatchment[0]
    Tlag = ParCatchment[1]
    frac = ParCatchment[2]
    frac1 = ParCatchment[3]
    k = ParCatchment[4]
    tmax = len(forcing)
    dt = 1

    #initialize states [Si, Su, Sf]
    States_plateau=np.zeros((tmax,3))#, columns=['Si', 'Su', 'Sf']) 
    States_hillslope=np.zeros((tmax,3))#, columns=['Si', 'Su', 'Sf'])
    States_hillslope_forest= np.zeros((tmax,3))#, columns=['Si', 'Su', 'Sf'])
    States_hillslope_crop= np.zeros((tmax,3))#, columns=['Si', 'Su', 'Sf'])
    States_wetland= np.zeros((tmax,3))#, columns=['Si', 'Su', 'Sf'])
    Ss=np.zeros((tmax,1))

    #initialize fluxes # [Eidt, Eadt, Qfdt, Qusdt]
    Fluxes_plateau= np.zeros((tmax,4))#, columns=['Eidt', 'Eadt', 'Qfdt','Qusdt' ])
    Fluxes_hillslope= np.zeros((tmax,4))#, columns=['Eidt', 'Eadt', 'Qfdt', 'Qusdt'])
    Fluxes_hillslope_forest= np.zeros((tmax,4))#, columns=['Eidt', 'Eadt', 'Qfdt', 'Qusdt' ])
    Fluxes_hillslope_crop=np.zeros((tmax,4))#, columns=['Eidt', 'Eadt', 'Qfdt', 'Qusdt'])
    Fluxes_wetland= np.zeros((tmax,4))#, columns=['Eidt', 'Eadt', 'Qfdt', 'Qusdt'])
    
    #initialize Conc [Cu, Cf, Cs]
    Conc_plateau = np.zeros((tmax, 2)) # columns=['Cu', 'Cf']
    Conc_hillslope = np.zeros((tmax, 2)) # columns=['Cu', 'Cf']
    Conc_hillslope_forest = np.zeros((tmax, 2)) # columns=['Cu', 'Cf']
    Conc_hillslope_crop = np.zeros((tmax, 2)) # columns=['Cu', 'Cf']
    Conc_wetland = np.zeros((tmax, 3)) # columns=['Cu', 'Cf', 'Cs']
    
    Qsdt = np.zeros(tmax)
    Qtotdt = np.zeros(tmax)
    
    Ctot = np.zeros(tmax)
    
    Ea = np.zeros(tmax)

    #loop over time
    for t in range(0,tmax):
        
        #plateau
        Fluxes_plateau, States_plateau, Conc_plateau = plateau(t, ParPlateau, forcing, Fluxes_plateau, States_plateau, Conc_plateau)
        
        #hillslope
        Fluxes_hillslope_forest, States_hillslope_forest, Conc_hillslope_forest = hillslope(t, ParHillslope_forest, forcing, Fluxes_hillslope_forest, States_hillslope_forest, Conc_hillslope_forest)
        Fluxes_hillslope_crop, States_hillslope_crop, Conc_hillslope_crop = hillslope(t, ParHillslope_crop, forcing, Fluxes_hillslope_crop, States_hillslope_crop, Conc_hillslope_crop)

        #wetland
        Fluxes_wetland, States_wetland, Ss, Conc_wetland = wetland(t, ParWetland, forcing, Fluxes_wetland, States_wetland, Conc_wetland, Ss, landscapes[2] )

        # Slow Reservoir
        Ss[t]=Ss[t]+((1-frac)*Fluxes_hillslope_forest[t, 3] + frac*Fluxes_hillslope_crop[t, 3])*dt*landscapes[1]+ Fluxes_wetland[t,3]*dt*landscapes[2]+ Fluxes_plateau[t,3]*dt*landscapes[0]

        Ea[t] = ((1-frac)*Fluxes_hillslope_forest[t,1]+frac*Fluxes_hillslope_crop[t,1])*landscapes[1]+Fluxes_plateau[t,1]*landscapes[0]+Fluxes_wetland[t,1]*landscapes[2]
        
        Ea_w =  Fluxes_wetland[t,1]*landscapes[2]
        
        Def = min(Ss[t], frac1*(forcing[t,2]-Ea_w))
        
        Ss[t] = Ss[t] - Def
        
        Ea[t] = Ea[t] + frac1 * Def
        
        Qsdt= dt*Ks*Ss[t] 
        Ss[t]=Ss[t]-min(Qsdt,Ss[t])
        if t<tmax-1:
            Ss[t+1]=Ss[t]
        
            
        # concentration in slow reservoirs
        if Ss[t] > 0:
            Conc_wetland[t,2] = (frac*Fluxes_hillslope_crop[t, 3]*dt*landscapes[1]*Conc_hillslope_crop[t,0] + Fluxes_wetland[t,3]*dt*landscapes[2]*Conc_wetland[t,0] + Fluxes_plateau[t,3]*dt*landscapes[0]*Conc_plateau[t,0] - k*Ss[t]*Conc_wetland[t,2]) / Ss[t] + Conc_wetland[t,2]
        else:
            Conc_wetland[t,2] = 0
        
        Qtotdt[t]=Qsdt+((1-frac)*Fluxes_hillslope_forest[t,2] + frac*Fluxes_hillslope_crop[t,2])*landscapes[1]+Fluxes_plateau[t,2]*landscapes[0]+Fluxes_wetland[t,2]*landscapes[2]                      
        
                
        # concentration at the final discharge
        if Qtotdt[t] > 0:
            Ctot[t] = (Qsdt*Conc_wetland[t,2] + frac*Fluxes_hillslope_crop[t,2]*landscapes[1]*Conc_hillslope_crop[t,1] + Fluxes_plateau[t,2]*landscapes[0]*Conc_plateau[t,1] + Fluxes_wetland[t,2]*landscapes[2]*Conc_wetland[t,1]) / Qtotdt[t]
        else:
            Ctot[t] = 0
            
    # Offset Q

    Weigths=Weigfun(Tlag)
    Qm = np.convolve(Qtotdt,Weigths)
    Qm=Qm[0:tmax]
    
    return(Qm, Ea, Ctot)

    
    