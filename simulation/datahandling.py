"""
data_handling.py

functions to load, preprocess, and maybe postprocess the data

"""

################################################################################################
# package imports
import sys
from casadi.tools import *
from datetime import datetime as dt
from datetime import timedelta


################################################################################################
# file imports
import config_sim as cfg

def get_time():
    """
        creates list of timesteps based on user set start and stop time of simulation
        :return:
        """
    timestep = cfg.start
    dt = []

    # get simulator timesteps
    timesteps = [timestep]
    while timestep < (cfg.stop - timedelta(minutes=5)):
        timestep += timedelta(minutes=5)
        timesteps.append(timestep)
    for ts in timesteps:
        delta = ts - timesteps[0]
        dt.append(delta.total_seconds() / 60)

    # get simulation + prediction horizon timesteps

    timestep = cfg.start
    timesteps_N = [timestep]
    dt_N = []
    while timestep < (cfg.stop + timedelta(hours=cfg.n_mpc) - timedelta(minutes=5)):
        timestep += timedelta(minutes=5)
        timesteps_N.append(timestep)

    for ts in timesteps_N:
        delta = ts - timesteps_N[0]
        dt_N.append(delta.total_seconds() / 60)

    return timesteps, timesteps_N, dt, dt_N

def get_spot_data():
    pass

def get_mpc_data(time):
    pass
def get_outside_temperature():

    pass
