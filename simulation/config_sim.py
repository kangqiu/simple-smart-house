"""
config_sim.py

configuration file for simple house simulator

- would be good if you could also plot from saved results file -> different script?
- add varying desired temperature and minimum temperature values
"""


################################################################################################
# package imports
from casadi.tools import *
import pytz
import datetime as dt
import numpy as np


################################################################################################
# file imports
import datahandling
################################################################################################

# data
local_timezone = pytz.timezone('Europe/Oslo')
n_mpc = 288 #24 hour prediction window
start = dt.datetime(2022, 1, 1, 0, 0).astimezone(local_timezone)
stop = dt.datetime(2022, 1, 10, 0, 0).astimezone(local_timezone)
# outside temperature data
temp_file = './data/SEKLIMAData_2022.pkl'
# spot data
spot_file = './data/SpotData2022_Trheim.pkl'
# noise file
noise_file = "./data/noise.pkl"

#set outside temperature and spot price constant for now
t_out_data = [5] * 288 * (7+1)
spot_price_data = [22] * 288 * (7+1)



################################################################################################
# references
t_desired = 21
t_min = 17

################################################################################################
# simulation config
ts = 5 # 5 min sampling time
add_noise = True
history = [{
    'room': 19.0,
    'wall': 18.3,
    'target': 18,
    'room_noise': 0,
    'power_noise': 0
}]

################################################################################################
# simulation model equations
""" we can add a power prediction, but we can also just set the control action to be the power directly"""
# control of heat pump power
t_target = MX.sym('t_target')
t_wall = MX.sym('t_wall') #outside temperature
t_room = MX.sym('t_room') #
t_out = MX.sym('t_out')

# TODO: add disturbance before saturation function
k = 0.2
ReLu = 100
maxpow =  1.5
pow = k * (t_target - t_room)

power = Function(
    'power', [t_room, t_target], [pow]
)

unsatpower = MX.sym('unsatpower')
powsat = log(1 + exp(ReLu * unsatpower)) / ReLu
hppow = powsat - log(1 + exp(ReLu * (powsat - maxpow))) / ReLu
satpower = Function('satpower', [unsatpower], [hppow])

#temperature prediction model
# m_air = 31.02679204362912 + 1
# m_wall = 67.21826736655125 + 1
# rho_in = 0.36409940390361406 - 0.1
# rho_out = 0.03348756113438382 - 0.015
# rho_dir = 0.03348756179891388 - 0.01

COP = 3.0
# rho_out_wall = 2.7504e-4
# rho_in_wall = 3.9289e-3
rho_out_wall = 4.9819e-4
rho_in_wall = 5.416673e-3

t_wall_plus =  t_wall + rho_out_wall * (t_out - t_wall) + rho_in_wall * (t_room - t_wall)
wallplus = Function('wallplus', [t_wall, t_room, t_out], [t_wall_plus])

pow = MX.sym('pow')
# rho_in_room = 8.52296e-3
rho_in_room = 0.0117
# rho_dir = 7.570089979e-4
rho_dir = 1.0793e-3

t_room_plus = t_room + rho_in_room * (t_wall - t_room) + rho_dir * (t_out - t_room) + COP * pow

roomplus = Function(
    "roomplus", [t_wall, t_room, t_out, pow], [t_room_plus]
)


################################################################################################
# mpc config

solver_options = {
    "linear_solver": "ma27"
}  # leave empty if you don't have the HSL solver library installed (highly recommended!)

################################################################################################
# mpc equations

""" so far the same as the simulation equations"""

################################################################################################
# noise parameters
noise = {   'mu' : { 'room': 0, 'power': 0.033},
            'sig': { 'room': 0.005317236877233841, 'power': 0.2963125110789097},
            'beta': { 'room': 0.99, 'power': 0.92},
            'epsilon': { 'room': 0.73, 'power':  0.68},
}

