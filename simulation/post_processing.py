"""
post_processing.py

plot results, calculates closed loop performance
"""
################################################################################################
# package imports
from casadi.tools import *
import pandas as pd
import pickle as pkl
from matplotlib import pyplot as plt

################################################################################################
# file imports
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)

import datahandling
import config_sim as cfg

#read results file
# results_file = '../results/01_allnoise_basecase.pkl'
results_file = '../results/September_tuned_derandomizev1.pkl'
print(results_file)
df_history = pkl.load(open(os.path.join(results_file), 'rb'))

#plot
timesteps, timesteps_N, dt, dt_N = datahandling.get_time()
# datahandling.plot(df_history, 0, len(timesteps), savefig = True)
datahandling.plot(df_history, 0, len(timesteps))

#calculate closed loop costs
# calculate closed loop cost:
J = 0

comfort_cost = 0
actuation_cost = 0
spot_cost = 0

l_base = []
stop = len(df_history)-cfg.n_mpc-1
for ts in range(0, len(df_history)-cfg.n_mpc):
    l_comfort = 0
    # l_actuation = 0
    l_spot = 0

    l_comfort += (cfg.w_mid * (df_history['t_max'][ts] - df_history['room'][ts]) ** 2)
    l_comfort += (cfg.w_max* max(df_history['room'][ts] - df_history['t_desired'][ts], 0) ** 2)
    l_comfort += (cfg.w_min * max(df_history['t_min'][ts] - df_history['room'][ts], 0) ** 2)


    # l_actuation += (cfg.w_target * (cfg.hubber ** 2) * (sqrt(1 + (df_history['target'][ts] -
    #                 df_history['target'][ts-1] / cfg.hubber) ** 2) - 1))

    l_spot += cfg.w_spot * df_history['spot_price'][ts] * df_history['power'][ts]

    comfort_cost += l_comfort
    # actuation_cost += l_actuation
    spot_cost += l_spot

    # J += l_comfort + l_actuation + l_spot
    J += l_comfort + l_spot
    l_base.append( l_comfort + l_spot)


plt.subplot(2, 1, 1)
plt.plot(range(0, stop), df_history['t_max'].values.tolist()[0: stop], label='t_desired')
plt.plot(range(0, stop), df_history['t_min'].values.tolist()[0: stop], label='t_min')
plt.plot(range(0, stop), df_history['room'].values.tolist()[0: stop], label='t_room')
plt.legend()
plt.grid("on")

plt.subplot(2, 1, 2)
plt.plot(range(0, stop), l_base[0: stop], label='closed loop cost')
plt.legend()
plt.grid("on")

plt.show()



print("closed loop cost", J)
print("electricity cost", spot_cost)
print("comfort cost", comfort_cost)

