"""
post_processing.py

plot results, calculates closed loop performance
"""
################################################################################################
# package imports
from casadi.tools import *
import pandas as pd
import pickle as pkl

################################################################################################
# file imports
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)

import datahandling
import config_sim as cfg

#read results file
results_file = './results/newsat/01_training_january.pkl'
df_history = pkl.load(open(os.path.join(results_file), 'rb'))

#plot
timesteps, timesteps_N, dt, dt_N = datahandling.get_time()
datahandling.plot(df_history, 0, len(timesteps))

#calculate closed loop costs
# calculate closed loop cost:
J = 0

comfort_cost = 0
actuation_cost = 0
spot_cost = 0

for ts in range(1, len(df_history)-cfg.n_mpc*2):
    l_comfort = 0
    l_actuation = 0
    l_spot = 0

    l_comfort += (cfg.w_tabove * (df_history['t_desired'][ts] - df_history['room'][ts]) ** 2)
    l_comfort += (cfg.w_tbelow * max(df_history['room'][ts] - df_history['t_desired'][ts], 0) ** 2)
    l_comfort += (cfg.w_tmin * max(df_history['t_min'][ts] - df_history['room'][ts], 0) ** 2)


    l_actuation += (cfg.w_target * (cfg.hubber ** 2) * (sqrt(1 + (df_history['target'][ts] -
                    df_history['target'][ts-1] / cfg.hubber) ** 2) - 1))

    l_spot += cfg.w_spot * df_history['spot_price'][ts] * df_history['power'][ts]

    comfort_cost += l_comfort
    actuation_cost += l_actuation
    spot_cost += l_spot

    J += l_comfort + l_actuation + l_spot

print("closed loop cost", J)
print("electricity cost", spot_cost)
print("comfort cost", comfort_cost)
print("actuation cost", actuation_cost)