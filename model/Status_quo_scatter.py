import numpy       as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import glob, os

save_loc = r'{}/output'.format(os.path.dirname(os.getcwd()))

# go to locations where files are saved
os.chdir(save_loc)
scenario = []
FSR = []
production = []
eta = []
U_max = []

# read all files that end with _sq to get all the status quo files
# open these and save the FSR, yield and scenario name into lists
for file in glob.glob('*_sq.csv'):
    scenario_name = file[:-7]
    df_sc = pd.read_csv(file)
    scenario.append(scenario_name)
    FSR.append(df_sc['y_star'][0])
    production.append(df_sc['x_star'][0])
    eta.append(df_sc['eta_used'][0])
    U_max.append(df_sc['U_max'][0])

# turn lists into dictionary
sq_dict = {'Scenario':scenario,
           'FSR (-)':FSR,
           'Production (RS)': production,
           'eta':eta,
           'U_max':U_max}

# turn dictionary into dataframe
df_sq = pd.DataFrame(data=sq_dict)    

# make scatter plot
fig, ax = plt.subplots(figsize=(8,8))
df_sq.plot.scatter(x='Production (RS)', y='FSR (-)', ax=ax, marker='X', color='tab:red')

# add annotation
for idx, row in df_sq.iterrows():
    ax.annotate(f'{row.Scenario} \nU: {row.U_max:.3f} (-)', (row['Production (RS)'], row['FSR (-)']))

# blabla
ax.set_ylim(13.9, 14.5)
ax.set_xlim(2.8*(10**10), 3.8*(10**10))
ax.set_title('Status Quo Values For All Scenarios')
ax.grid()
plt.tight_layout()
plt.savefig(r'{}\status_quo_points.png'.format(save_loc))

