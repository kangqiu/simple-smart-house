"""
config_sim.py

configuration file for simple house simulator

- get noise trajectories
- implement some rudimentary plotting
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
stop = dt.datetime(2022, 1, 7, 0, 0).astimezone(local_timezone)
# outside temperature data
temp_file = './data/SEKLIMAData_2022.pkl'
# spot data
spot_file = './data/SpotData2022_Trheim.pkl'

#set outside temperature and spot price constant for now
t_out_data = [5] * 288 * (7+1)
spot_price_data = [22] * 288 * (7+1)



################################################################################################
# parameters
m_air = 31.02679204362912 + 1
m_wall = 67.21826736655125 + 1
rho_in = 0.36409940390361406 - 0.1
rho_out = 0.03348756113438382 - 0.015
rho_dir = 0.03348756179891388 - 0.01
fan = 0.2430419894184425

t_desired = 21
t_min = 17

hpset = 0.06071641164806341 + 0.04
hpin = 0.08436238305547132  - 0.02
hp0 = 1.1441609877948857 
ReLu = 100
maxpow =  3

################################################################################################
# simulation config
ts = 5 # 5 min sampling time
n_sim = 288 * 7  # 24h * n simulation
history = [{
    'room': 19.0,
    'wall': 18.3,
    'noise': 0,
    'target': 18,
}]

################################################################################################
# simulation model equations
""" we can add a power prediction, but we can also just set the control action to be the power directly"""
# control of heat pump power
t_target = MX.sym('t_target')
t_wall = MX.sym('t_wall') #outside temperature
t_room = MX.sym('t_room') #
t_out = MX.sym('t_out')

DT = hpset * t_target
DT -= hpin * t_room
pow =  (DT + hp0)
powsat = log(1 + exp(ReLu * pow)) / ReLu
hppow = powsat - log(1 + exp(ReLu * (powsat - maxpow))) / ReLu

power = Function(
    "power", [t_room, t_target], [hppow]
)

# define prediction model

# I had to set the COP very very high, dunno how to fix that without having to fit a heat pump model
COP = 3.0 
t_wall_plus = m_wall * t_wall + rho_out * (t_out - t_wall) + rho_in * (t_room - t_wall)
t_wall_plus *= 1 / m_wall
wallplus = Function(
    "wallplus", [t_wall, t_room, t_out], [t_wall_plus]
)

pow = MX.sym('pow')

t_room_plus = (
    m_air * t_room
    + rho_in * (t_wall - t_room)
    + rho_dir * (t_out - t_room)
    + COP * pow
)
t_room_plus *=  1 / m_air

roomplus = Function(
    "roomplus", [t_wall, t_room, t_out, pow], [t_room_plus]
)


################################################################################################
# mpc config

#weights
w_spot = 0.1 #weight spot cost
w_tbelow = 0.2 #weight temperature below
w_tabove = 0.005 #weight temperature above
w_tmin = 50
w_target = 0.5

solver_options = {
    "linear_solver": "ma27"
}  # leave empty if you don't have the HSL solver library installed (highly recommended!)

################################################################################################
# mpc equations

""" so far the same as the simulation equations"""


