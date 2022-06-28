                                                                                                                                                                                                                                                        
import numpy       as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from pyfunctions.FLEXtopo import FLEXtopo
from scipy.interpolate import interp1d

def agriculture_production(frac, frac1, extra_params, frac2=0.75): 
    '''    
    Function to calculate the agricultural production
    returns total production, modelled discharge and concentrations
    frac: percentage of crop land on hillslope
    frac1: percentage of irrigation on wetland
    frac2: percentage of agriculture area on plateau (0.75=basic)
    '''

    compo, Ky, Yo, profit, start_irrigating, end_irrigating, area_Plateau, area_Hillslope, area_Wetland, landscape_per, ParPlateau, ParHillslope_forest, ParHillslope_crop, ParWetland, ParCatchment, forcing = extra_params.values()
    
    # add frac and frac1 to the catchment parameters
    ParCatchment[2] = frac
    ParCatchment[3] = frac1
    
    Qm, Ea, Ctot = FLEXtopo(ParPlateau, ParHillslope_forest, ParHillslope_crop, ParWetland, ParCatchment, forcing, landscape_per)
    
    Ep = forcing[:,2]
 
    # calculate total Ea for crops in 2013
    sum_Ea = np.sum(Ea)  
    
    # make list with all required ETa and Ep
    N_crops = len(start_irrigating)
    ETa_list = np.zeros(N_crops)
    Ep_list = np.zeros(N_crops)
    for i in range(N_crops):
        start = start_irrigating[i]
        end = end_irrigating[i]
        sum_Ea = np.sum(Ea[start:end])
        sum_Ep = np.sum(Ep[start:end])
        ETa_list[i] = sum_Ea
        Ep_list[i] = sum_Ep
    
    # get total irrigated and rainfed area
    area_irrigation = area_Wetland * frac1
    area_rainfed = area_Plateau * frac2 + area_Hillslope * frac + area_Wetland * (1 - frac1)
 
    # yield rainfed area
    Ya = (1 - Ky * (1 - ETa_list / Ep_list)) * Yo
    
    # total production of rainfed and irrigated area
    Production = (Ya * area_rainfed * compo + Yo * area_irrigation * compo) * profit
    
    # calculate total production
    tot_production = np.sum(Production)
        
    return tot_production, Qm, Ctot


def species(Qm, Ctot, beta0=15, beta1=0.25, c_threshold=1):
    '''
    Species richness function.
    Calculates Es.
    Qm: Modelled discharge
    Ctot: total concentration
    beta0: 15 (by itself, from literature)
    beta1: 0.25 (by itself, from literature)
    c_threshold: fish species richness, by default 1 mg/m3 = 0.1 kg/(mm*ha) 
    '''
    
    # define  Qgrey and Es array
    Qgrey = np.zeros(len(Qm))
    Es = np.zeros(len(Qm))
    
    # iterate over modelled Q and calculate Es
    for i in range(0, len(Qm)):
        
        Qgrey[i] = max(0, Ctot[i] / c_threshold - 1) * Qm[i]
        Qgrey[i] = min(Qm[i], Qgrey[i])
        Es[i] = beta0 * (Qm[i] - Qgrey[i]) ** beta1
        
    # average Es
    Es_avg = np.average(Es)
    
    return Es_avg


def remove_non_concave_pts(Es_avg, yields, frac1s):
    '''
    Function that removes non-concave points
    '''
    
    # derive points two times
    dEdY = np.zeros_like(Es_avg)
    l = 2 # smoothing purposes!!
    for i in range(l, len(Es_avg)-l, 1):
        dEdY[i] = (Es_avg[i+l] - Es_avg[i-l]) / (2 * l)
    ddEdYY = np.zeros_like(dEdY)
    for i in range(1, len(Es_avg)-1,1):
        ddEdYY[i] = (dEdY[i+1] - dEdY[i-1]) / 2
    
    # remove garbage created at edges
    garbage_hyperparameter = 5
    dEdY[:garbage_hyperparameter] = np.nan
    dEdY[-garbage_hyperparameter:] = np.nan
    ddEdYY[:garbage_hyperparameter] = np.nan
    ddEdYY[-garbage_hyperparameter:] = np.nan
    
    # get lowest index of the double derivative
    i_min = np.argmin(ddEdYY[garbage_hyperparameter:-garbage_hyperparameter]) + garbage_hyperparameter
    i_left_bound = np.argmax(dEdY[garbage_hyperparameter:i_min]) + garbage_hyperparameter
    i_right_bound = i_min + np.argmax(ddEdYY[i_min:-garbage_hyperparameter])
        
    return Es_avg[i_left_bound:i_right_bound], yields[i_left_bound:i_right_bound], frac1s[i_left_bound:i_right_bound]


def status_quo_point(frac, frac1, p, extra_params):
    '''
    returns yield and Es (FSR) of the status quo situation located on the PPF-curve
    frac: percentage of crop land on hillslope 
    frac1: percentage of irrigation on wetland
    p: list with polynomials
    '''
    
    tot_production, Qm, Ctot = agriculture_production(frac, frac1, extra_params)
    yield_status_quo = tot_production
    Es_status_quo = p[0] * (yield_status_quo ** 2) + p[1] * yield_status_quo + p[2]
    
    return yield_status_quo, Es_status_quo


def calc_eta(yield_status_quo, Es_status_quo, p):
    '''
    Calculates Eta value
    yield_status_quo: yield status quo
    Es_status_quo: Es (FSR) status quo
    '''
    
    return -Es_status_quo / (2 * p[0] * yield_status_quo ** 2 + p[1] * yield_status_quo)


def max_utility(p, eta):
    '''
    p: list with three polynomials p0, p1, p2
    eta: eta value from status quo
    returns maximum U and x_star and y_star
    '''
    
    # Calculate maximum y_star and x_star, which is the optimum 
    # according to the lagrange derivation
    # use quadratic formula for x
    a = p[0] * (1 + 2 * eta)
    b = p[1] * (1 + eta)
    c = p[2]
    
    x_star = (-b - np.sqrt(b**2 - 4*a*c)) / (2 * a) # TAKE LOWER LIMIT
    y_star = p[0] * x_star ** 2 + p[1] * x_star + p[2]
    U_max = np.log(x_star) + eta * np.log(y_star)
    
    return U_max, x_star, y_star


def utility_curve(x, U, eta):
    """
    Creates utility curve y-data
    x: x data
    U: value of utility to fit curve on (mostly the maximum utility U(xstar, ystar))
    eta: eta value, eta mostly taken from status quo
    """

    y = np.exp((U - np.log(x)) / eta)
    
    return y


def ppf_curve(frac, extra_params, dfrac1=0.01):
    '''
    calculates points to plot ppf curve with, 
    returns Es and Yield for different frac1 values
    frac: frac value
    dfrac: step dfrac1s
    '''
    
    # make frac1 values to iterate through for every x-star on the graph
    frac1s = np.arange(dfrac1, 1, dfrac1)
    
    # define Es and yields arrays
    Es_avg = np.zeros_like(frac1s)
    yields = np.zeros_like(frac1s)
    
    # fill Es and yields arrays with the FSR and agricultural production
    for i, fraction in enumerate(frac1s):
        tot_production, Qm, Ctot = agriculture_production(frac, fraction, extra_params)
        Es_avg[i] = species(Qm, Ctot)
        yields[i] = tot_production
        
    return Es_avg, yields, frac1s


def hydro_economic_analysis(environmental_data_params, frac_status_quo, 
                            frac1_status_quo, extra_params, 
                            MODELNAME, save_loc):
    '''
    Performs some hydro economic magic
    frac_status_quo: frac status quo
    frac1_status_quo: frac1 status quo
    extra params: dictionary with extra params for flextopo and agricultural production
    MODELNAME: name model
    save_loc: location to save data csv files and plots
    '''
    
    # unpack required variables
    xdata, P, Ea, Ep, Q, Qm, Ctot, Wstart, eta = environmental_data_params.values()
    
    # get Es (FSR) and yields (Production) of points
    # remove non concave points of these as well
    Es_avg_nc, yields_nc, frac1s = ppf_curve(frac_status_quo, extra_params, dfrac1=0.01)
    Es_avg, yields, frac1s = remove_non_concave_pts(Es_avg_nc, yields_nc, frac1s)
    
    # get required polynomials
    p = np.polyfit(yields, Es_avg, 2)

    # get status quo situation on ppf curve
    yield_status_quo, Es_status_quo = status_quo_point(frac_status_quo, frac1_status_quo, p, extra_params)
        
    # make utility curve
    x_fit = np.linspace(yields[0] * 0.75, yields[-1] * 1.25, 10000)
    
    # if eta is not defined then calculate a new eta to use for the utility 
    # curve. Otherwise calculate a new eta value to save with the status quo
    # values
    eta_scenario = calc_eta(yield_status_quo, Es_status_quo, p)
    if eta == None:
        print('Calculated eta for scenario is {}'.format(eta_scenario))
        eta = eta_scenario
    
    # get corresponding U for status quo situation
    U_sq = np.log(yield_status_quo) + eta * np.log(Es_status_quo)    
    
    # calculate the corresponding frac1 value from the maximization point
    # calculate corresponding frac1 value of model
    # calculate at Production points from model the FSR via polynomails
    Es_PPF_pts = p[0] * yields ** 2 + p[1] * yields + p[2]
    
    # calculate U for these points
    U_pts = np.log(yields) + eta * np.log(Es_PPF_pts)
    
    # calculate optimum for U (U_max) with their x and y coordinates
    U_max, x_star, y_star = max_utility(p, eta)
    
    # calculate corresponding frac for the U_max using linear interpolation
    i_max_U = np.argmax(U_pts)
    frac_optimized =frac1s[i_max_U]
    
    # calculate utility curve
    y_utility = utility_curve(x_fit, U_max, eta)
    
    # calculate polynomial curve on the PPF points which results in the PPF
    # curve
    Es_fit_ppf = p[0] * x_fit ** 2 + p[1] * x_fit + p[2]

    def transform_line_bar_data(x, y):
        '''
        Magic.
        '''
        ys = []
        xs = x.repeat(3)
        for i in y:
            ys.append(None)
            ys.append(0)
            ys.append(i)
                
        ys = np.array(ys)
        return ys, xs
    
    
    if save_loc != None:
    
        # make plot for PPF
        plt.figure(figsize=(8,6))
        plt.title('{}: PPF and Utility Curve, $\eta$ = {}'.format(MODELNAME, np.round(eta,3)))
        plt.plot(yields_nc, Es_avg_nc, '.', label='PPF point')        
        plt.plot(yields, Es_avg, '.', label='Concave PPF point')
        plt.plot(x_fit, Es_fit_ppf, label='PPF Curve')
        plt.plot(yield_status_quo, Es_status_quo, 'X', label=f'Status Quo point, U: {U_sq:.3f} (-), frac1: {frac1_status_quo:.2f} (-)', color='tab:green')
        plt.plot(x_fit, y_utility, label='Utility Curve')
        plt.plot(x_star, y_star, 'X', label=f'Optimum Utility Curve, U: {U_max:.3f} (-), frac1: {frac_optimized:.2f} (-)', color='tab:red')
        plt.ylim(10.5, 17)
        plt.xlim(1*(10**10), 7*(10**10))
        plt.ylabel('FSR (-)')
        plt.xlabel('Production (RS)')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(r'{}_PPF_Utility.png'.format(save_loc))
    
        # plot discharge and fertilizer
        fig, ax0 = plt.subplots(figsize=(15,7.5))
    
        # make different axes
        ax01 = ax0.twinx()
        ax02 = ax0.twinx()
        ax02.spines.right.set_position(("axes", 1.05))
        ax02.invert_yaxis()
    
        # plot on axis
        l0 = ax0.plot(xdata, Q, label='Measured Discharge')
        l1 = ax0.plot(xdata, Qm, label='Modelled Discharge')
        l2 = ax01.plot(xdata, Ctot,'--', label='Modelled concentration fertilizer in water', color='red')
        Ws, xd, = transform_line_bar_data(xdata, Wstart.reshape(-1))
        l3 = ax02.plot(xd, Ws, label='Fertilizer', color='purple')
        ax02.set_ylim(600, 0)
        ax0.set_ylim(-1, 80)
        ax01.set_ylim(-1*16/80, 16)
    
        # add labels
        ax01.set_ylabel('Concentration kg/(mm*ha)')
        ax02.set_ylabel('Fertilizer added (kg)')
        ax0.set_ylabel('Discharge (m3/d)')
        ax0.set_xlabel('Day of year (d)')
        ax0.set_title('{}: Discharge and Concentration'.format(MODELNAME))
    
        # legend
        lt0 = l0 + l1 + l2 + l3
        ax0.legend(lt0, [l.get_label() for l in lt0], loc='center right')
        fig.tight_layout()
        plt.savefig(r'{}_Q_C.png'.format(save_loc))
    
        # plot precipitation and evaporation
        fig, ax1 = plt.subplots(figsize=(15,7.5))
    
        # make different axes
        ax11 = ax1.twinx()
        ax11.invert_yaxis()
    
        # plot
        l3 = ax1.plot(xdata, Ea, label='Modelled Actual Evaporation', color='limegreen')
        l4 = ax1.plot(xdata, Ep, label='Potential Evaporation', color='darkgreen')
        Ps, xs = transform_line_bar_data(xdata, P)
        l5 = ax11.plot(xs, Ps, label='Measured Precipitation')
        ax1.set_ylim(-0.1, 8.5)
        ax11.set_ylim(250, 0)
    
        # labels
        ax1.set_ylabel('Evaporation (mm/d)')
        ax11.set_ylabel('Precipitation (mm/d)')
        ax1.set_xlabel('Day of year (d)')
    
        # legend
        lt1 = l3 + l4 + l5
        ax1.legend(lt1, [l.get_label() for l in lt1], loc='lower right')
        ax1.set_title('{}: Precipitation and Evaporation'.format(MODELNAME))
        fig.tight_layout()
        plt.savefig(r'{}_PCP_ET.png'.format(save_loc))
        
        # Make plot for frac1 vs utiltiy to show that what we did works
        plt.figure(figsize=(8,8))
        plt.plot(frac1s, U_pts, '.', color='tab:orange', label='Utility per model run of Frac1')
        plt.plot(frac_optimized, U_max, 'X', color='tab:red', label='Optimal Utility')
        plt.xlabel('Frac1 values (-)')
        plt.ylabel('Utility (-)')
        plt.title(f'{MODELNAME}: Utility vs Frac1, U: {U_max:.3f} (-), corresponding frac1: {frac_optimized:.2f} (-).')
        plt.legend()
        plt.tight_layout()
        plt.savefig(r'{}_U_vs_F.png'.format(save_loc))
    
        # add data in dataframe in order to save it
        # make dictionary to put in dataframe and save :)
        dict_ppf_data = {'x_fit':x_fit,
                         'Es_fit_ppf':Es_fit_ppf,
                         'y_utility':y_utility}
        
        dict_FSR_Production = {'FSR_pts': Es_avg,
                               'Production_pts':yields,
                               'Frac1s': frac1s}
    
        dict_environmental_data = {'xdata':xdata,
                                   'Q':Q,
                                   'Qm':Qm,
                                   'Wstart':Wstart.T[0],
                                   'Ctot': Ctot,
                                   'P': P,
                                   'Ep': Ep,
                                   'Ea': Ea}

        dict_status_quo = {'frac_status_quo': [frac_status_quo],
                           'frac1_status_quo': [frac1_status_quo],
                           'yield_status_quo': [yield_status_quo],
                           'Es_status_quo': [Es_status_quo],
                           'eta_used': [eta],
                           'eta_scenario': [eta_scenario],
                           'x_star':[x_star],
                           'y_star':[y_star],
                           'U_max': [U_max],
                           'frac1_max': [frac_optimized],
                           'U_sq': [U_sq]}
    
        # turn dictionaries into dataframe and save
        df_ppf = pd.DataFrame(data=dict_ppf_data)
        df_ppf.to_csv(r'{}_ppf.csv'.format(save_loc))
        df_env = pd.DataFrame(data=dict_environmental_data)
        df_env.to_csv(r'{}_env.csv'.format(save_loc))
        df_sq = pd.DataFrame(data=dict_status_quo)
        df_sq.to_csv(r'{}_sq.csv'.format(save_loc))

    return yield_status_quo, Es_status_quo