import numpy       as np
from pyfunctions.Weigfun import Weigfun

def wetland(  timestep, Par, forcing, Fluxes, States, Conc, Ss, landscape_per ):
    #HBVpareto Calculates values of 3 objective functions for HBV model

    Imax=Par[0]
    Ce=Par[1]
    Sumax=Par[2]
    beta=Par[3]
    Cmax=Par[4]
    Kf=Par[5]
    k = Par[6] # decay factor

    Qo=forcing[:,0]
    Prec=forcing[:,1]
    Etp=forcing[:,2]
    W=forcing[:,3] # fertilizer injection, kg/ha

    tmax=len(Prec)
    Si=States[:,0]
    Su=States[:,1]
    Sf=States[:,2]

    Eidt=Fluxes[:,0]
    Eadt=Fluxes[:,1]
    Qfdt=Fluxes[:,2]
    
    Cu=Conc[:,0] # kg/(mm*ha)
    Cf=Conc[:,1] # kg/(mm*ha)
    Cs=Conc[:,2] # kg/(mm*ha)
    
    dt=1
    t=timestep

    Pdt=Prec[t]*dt
    Epdt=Etp[t]*dt
    
    # Interception Reservoir
    if Pdt>0:
        Si[t]=Si[t]+Pdt
        Pedt=max(0,Si[t]-Imax)
        Si[t]=Si[t]-Pedt
        Eidt[t]=0
    else:
        
    # Evaporation only when there is no rainfall
        Pedt=0
        Eidt[t]=min(Epdt,Si[t])
        Si[t]=Si[t]-Eidt[t]

    if t<tmax-1:
        Si[t+1]=Si[t]

    # Unsaturated Reservoir
    if Pedt>0:
        rho=(Su[t]/Sumax)**beta            
        Su[t]=Su[t]+(1-rho)*Pedt
        Qufdt=rho*Pedt
    else:
        Qufdt=0
    
    # Transpiration
    Epdt=max(0,Epdt-Eidt[t])
    Eadt[t]=Epdt*(Su[t]/(Sumax*Ce))
    Eadt[t]=min(Eadt[t],Su[t])
    Su[t]=Su[t]-Eadt[t]

    #Capillary rise
    Qrdt=(1-Su[t]/Sumax)*Cmax*dt;
    Qrdt=min(Qrdt, Ss[t]/landscape_per);

    if( (Su[t] + Qrdt) > Sumax):
        Qrdt = Sumax - Su[t]

    Su[t] = Su[t] + Qrdt
    Ss[t]=Ss[t]-Qrdt*landscape_per

    if t<tmax-1:
        Su[t+1]=Su[t]

    # concentration in the unsaturated reservoir
    if Su[t] > 0:
        Cu[t] = (Qrdt * Cs[t] + W[t] - Qufdt * Cu[t] - k * Cu[t] * Su[t]) / Su[t] + Cu[t]
    else:
        Cu[t] = 0        
        
    # Fast Reservoir
    Sf[t]=Sf[t]+Qufdt
    Qfdt[t]= dt*Kf*Sf[t]
    Sf[t]=Sf[t]-min(Qfdt[t],Sf[t])
    if t<tmax-1:
        Sf[t+1]=Sf[t]    

    # concentration in the fast reservoir
    if Sf[t] > 0:
        Cf[t] = (Cu[t] * Qufdt - Cf[t] * Qfdt[t] - k * Cf[t] * Sf[t]) / Sf[t] + Cf[t]
    else:
        Cf[t] = 0

    #save output
    States[:,0]=Si
    States[:,1]=Su
    States[:,2]=Sf

    Fluxes[:,0]=Eidt
    Fluxes[:,1]=Eadt
    Fluxes[:,2]=Qfdt
    
    Conc[:,0]=Cu #
    Conc[:,1]=Cf #
    Conc[:,2]=Cs #


    return(Fluxes, States, Ss, Conc)


