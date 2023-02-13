"""
main.py

main file for simple smart house mpc example
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
df_history = pkl.load(open(os.path.join(cfg.results_file), 'rb'))

#plot
timesteps, timesteps_N, dt, dt_N = datahandling.get_time()
datahandling.plot(df_history, 0, len(timesteps))
