import numpy       as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import glob, os

# save loc
save_loc = r'{}/output'.format(os.path.dirname(os.getcwd()))

# go to locations where files are saved
os.chdir(save_loc)

def plot_ppf_u(first_scenario, second_scenarios, save_loc):
    '''
    first scenario: scenario to keep plotting for comparison
    second_scenarios: list with scenarios to iterate over and plot with 
    the first one
    '''
    
    df1 = pd.read_csv(f'{save_loc}\{first_scenario}_ppf.csv')
    df1.set_index('x_fit', inplace=True)
    df1_vals = pd.read_csv(f'{save_loc}\{first_scenario}_sq.csv')

    for i, scenario in enumerate(second_scenarios):
        
        # read dataframe
        df2 = pd.read_csv(f'{save_loc}\{scenario}_ppf.csv')
        df2.set_index('x_fit',inplace=True)
        df2_vals = pd.read_csv(f'{save_loc}\{scenario}_sq.csv')
        
        fig, ax = plt.subplots(figsize=(8,8))
        df1[['Es_fit_ppf','y_utility']].plot(ax=ax)
        df2[['Es_fit_ppf','y_utility']].plot(ax=ax)
        ax.plot(df1_vals.yield_status_quo, df1_vals.Es_status_quo, 'X', color='tab:blue')
        ax.plot(df2_vals.yield_status_quo, df2_vals.Es_status_quo, 'X', color='tab:green')
        ax.plot(df1_vals.x_star, df1_vals.y_star, 'X', color='tab:orange')
        ax.plot(df2_vals.x_star, df2_vals.y_star, 'X', color='tab:red')
        ax.grid()
        ax.set_ylim(10.5, 17)
        ax.set_xlim(1*(10**10), 7*(10**10))
        ax.set_ylabel('FSR (-)')
        ax.set_xlabel('Production (RS)')
        ax.set_title('Production Possibility Frontier and Utility Curve')
        ax.legend([f'{first_scenario} PPF Curve', f'{first_scenario} Utility Curve',
                   f'{scenario} PPF Curve', f'{scenario} Utility Curve',
                   f'{first_scenario} Status Quo Point, U: {df1_vals.U_sq[0]:.3f} (-), frac1: {df1_vals.frac1_status_quo[0]:.2f} (-)',
                   f'{scenario} Status Quo Point, U: {df2_vals.U_sq[0]:.3f} (-), frac1: {df2_vals.frac1_status_quo[0]:.2f} (-)',
                   f'{first_scenario} Maximum Utility Point, U: {df1_vals.U_max[0]:.3f} (-), frac1: {df1_vals.frac1_max[0]:.2f} (-)', 
                   f'{scenario} Maximum Utility Point, U: {df2_vals.U_max[0]:.3f} (-), frac1: {df2_vals.frac1_max[0]:.2f} (-)'])
        fig.tight_layout()
        plt.savefig(f'{save_loc}\{first_scenario}_{scenario}_ppf_utility.png')

# first scenario, mostly reference scenario
first_scenario = 'Reference Scenario'
second_scenarios = ['Scenario 1', 'Scenario 2', 'Scenario 3', 'Scenario 4'] 

plot_ppf_u(first_scenario, second_scenarios, save_loc)

first_scenario = 'Status Quo Scenario'
second_scenarios = ['Reference Scenario', 'Scenario 1'] 

plot_ppf_u(first_scenario, second_scenarios, save_loc)
