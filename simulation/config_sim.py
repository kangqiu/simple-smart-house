"""
config_sim.py

configuration file for
"""


################################################################################################
# package imports
from casadi.tools import *
import pytz
import datetime as dt


################################################################################################
# file imports
import datahandling
################################################################################################

# data
local_timezone = pytz.timezone('Europe/Oslo')
n_mpc = 288 #24 hour prediction window
start = dt.datetime(2022, 11, 10, 0, 0).astimezone(local_timezone)
stop = dt.datetime(2022, 11, 17, 0, 0).astimezone(local_timezone)
# outside temperature data
temp_file = '../data/SEKLIMAData_2022.pkl'
# spot data
spot_file = '../data/SpotData2022_Trheim.pkl'

#set outside temperature and spot price constant for now
t_out_data = [5] * 288 * (7+1)
spot_price_data = [22] * 288 * (7+1)



################################################################################################
# parameters
m_air = 25.271 # 23.271
m_wall = 50.413
rho_in = 0.310
rho_out = 0.019
rho_dir = 0.019
GHI = 1.093

t_desired = 21
t_min = 17


################################################################################################
# simulation config
ts = 5 # 5 min sampling time
n_sim = 288 * 7  # 24h * n simulation
history = [{
    'room': 19.0,
    'wall': 18.3,
    'hp': 0.2,
    'noise': 0
}]

################################################################################################
# simulation model equations
""" we can add a power prediction, but we can also just set the control action to be the power directly"""
# define prediction model
t_wall = MX.sym('t_wall') #outside temperature
t_room = MX.sym('t_room') #
t_out = MX.sym('t_out')
hp = MX.sym('hp')

# COP is calculated with 0.2* the inverse carnot efficiency (theoretical efficiency of heat pumps)
# (desired temperature at home cold temp outside)
t_avg = -7  #in ˚C
COP = 0.2*((t_desired+273.15)/(t_desired - t_avg)) # roughly translates to a COP of 2.1

# Wall Temperature
t_wall_plus = m_wall * t_wall + rho_out * (t_out - t_wall) + rho_in * (t_room - t_wall)
wallplus = Function(
    "wallplus", [t_wall, t_room, t_out], [t_wall_plus]
)

t_room_plus = (
    m_air * t_room
    + rho_in * (t_wall - t_room)
    + rho_dir * (t_out - t_room)
    + COP * hp
)
roomplus = Function(
    "roomplus", [t_wall, t_room, t_out, hp], [t_room_plus]
)

################################################################################################
# mpc config

#weights
w_spot = 0.1 #weight spot cost
w_tbelow = 0.2 #weight temperature below
w_tabove = 0.005 #weight temperature above
w_tmin = 50
w_hp = 0.01 #actuation penalty

solver_options = {
    "linear_solver": "ma27"
}  # leave empty if you don't have the HSL solver library installed (highly recommended!)

################################################################################################
# mpc equations

""" so far the same as the simulation equations"""


