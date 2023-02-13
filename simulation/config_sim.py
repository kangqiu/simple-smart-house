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
# TODO: fix weights and simulate 3 days
# TODO: delete noise and see how it performs
# TODO: run an entire month of January
# data
local_timezone = pytz.timezone('Europe/Oslo')
n_mpc = 288 #24 hour prediction window
start = dt.datetime(2022, 1, 1, 0, 0).astimezone(local_timezone)
stop = dt.datetime(2022, 2, 1, 0, 0).astimezone(local_timezone)
# outside temperature data
temp_file = './data/SEKLIMAData_2022.pkl'
spot_file = './data/SpotData2022_Trheim.pkl'
noise_file = './data/noise_january_v2.pkl'
add_noise = True
# results are saved in this file
results_file = './results/simulation_january_v3.pkl'


################################################################################################

################################################################################################
# simulation config
ts = 5 # 5 min sampling time

################################################################################################
# simulation model equations
# control of heat pump power
t_target = MX.sym('t_target')
t_wall = MX.sym('t_wall')
t_room = MX.sym('t_room')
t_out = MX.sym('t_out')

HP0 = 1.1441609877948857
HPin = 0.08436238305547132
HPset = 0.06071641164806341

k = 0.2
ReLu = 100
maxpow =  1.5
pow = k * (t_target - t_room)
# pow = HPset * t_target - HPin * t_room + HP0

power = Function(
    'power', [t_room, t_target], [pow]
)

unsatpower = MX.sym('unsatpower')
powsat = log(1 + exp(ReLu * unsatpower)) / ReLu
hppow = powsat - log(1 + exp(ReLu * (powsat - maxpow))) / ReLu
satpower = Function('satpower', [unsatpower], [hppow])

#temperature prediction model
m_air = 31.02679204362912 #+ 30
m_wall = 67.21826736655125 #+ 20
rho_in = 0.36409940390361406 #+ 1
rho_out = 0.03348756113438382 #+ 0.1
rho_dir = 0.03348756179891388 #+ 0.1

# COP = 3
# COP0, COP1, T0 = (4, 0.1, 10)
# COP = (COP0 - np.log(1 + np.exp(ReLu * COP1 * (T0 - t_out))) /ReLu)
# COP = Function('COP', [t_out], [COP])
COP = 3
# rho_out_wall = 4.9819e-4
# rho_in_wall = 5.416673e-3
rho_out_wall = rho_out/m_wall
rho_in_wall = rho_in/m_wall
t_wall_plus =  t_wall + rho_out_wall * (t_out - t_wall) + rho_in_wall * (t_room - t_wall)
wallplus = Function('wallplus', [t_wall, t_room, t_out], [t_wall_plus])

power_HP = MX.sym('power_HP')
# rho_in_room = 0.0117 + 0.1
# rho_dir = 1.0793e-3 + 1e-2
rho_in_room = rho_out/m_air
rho_dir = rho_dir/m_air

cop = MX.sym('cop')
t_room_plus = t_room + rho_in_room * (t_wall - t_room) + rho_dir * (t_out - t_room) + cop * power_HP * 1/m_air

roomplus = Function(
    "roomplus", [t_wall, t_room, t_out, power_HP, cop], [t_room_plus]
)


################################################################################################
# mpc config

solver_options = {
    "linear_solver": "ma57"
}  # leave empty if you don't have the HSL solver library installed (highly recommended!)

################################################################################################
# mpc equations
# objective function parameterization
thetal_num = np.array([0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
thetal = MX.sym('thetal', 6)
thetat_num = np.array([1,1,0])
thetat = MX.sym('thetat', 3)
# MPC model parameterization
thetam = MX.sym('theta_m', 10)

# MPC objective function
w_spot = 0.1  # weight spot cost
w_tbelow = 0.2# weight temperature below
w_tabove = 0.005  # weight temperature above
w_tmin = 50
w_target = 0.5
hubber = 0.5

# define symbolic variables
t_desired = MX.sym('t_desired')
dt_target = MX.sym('dt_target')
slack = MX.sym('slack')
slackmin = MX.sym('slackmin')
spot = MX.sym('spot')


#define symbolic parameterized stage cost function
l_mpc = 0
l_mpc += thetal[0]
l_mpc += (thetal[1] * w_tabove * (t_desired- t_room)**2 / float(n_mpc))
l_mpc += (thetal[2] * w_tbelow * slack ** 2 / float(n_mpc))
l_mpc += (thetal[3] * w_tmin * slackmin ** 2 / float(n_mpc))
l_mpc += (thetal[4] * w_target * (hubber ** 2) * (sqrt(1 + (dt_target / hubber) ** 2) - 1) / float(n_mpc))
l_mpc += (thetal[5] * w_spot * (spot * power_HP) / float(n_mpc))

lmpc_func = Function(
    'lmpc_func', [t_desired, t_room, slack ,slackmin, dt_target, power_HP, spot, thetal], [l_mpc]
)

#parameterized terminal cost function

t_mpc = 0
t_mpc += (thetat[0] * w_tbelow * slack ** 2 / float(n_mpc))
t_mpc += (thetat[1] * w_tmin * slackmin ** 2 / float(n_mpc))
t_mpc += thetat[2]

tmpc_func = Function( 'tmpc_func', [slack, slackmin, thetat], [t_mpc])

#parameterized model equations MPC
# pow_mpc= thetam[0] * HPset * thetam[1]* t_target - HPin * t_room + HP0 +thetam[2]
pow_mpc = thetam[0] * k * (t_target - t_room) + thetam[1]
power_mpc = Function('power_mpc', [t_room, t_target, thetam], [pow_mpc])

t_wall_mpc =  t_wall + thetam[3] *rho_out_wall * (t_out - t_wall) + thetam[4] * rho_in_wall * (t_room - t_wall) + thetam[5]
wall_mpc = Function('wall_mpc', [t_wall, t_room, t_out, thetam], [t_wall_mpc])

t_room_mpc = t_room + thetam[6]*rho_in_room * (t_wall - t_room) + thetam[7] *rho_dir * (t_out - t_room) + thetam[8] *\
             cop * power_HP * 1/m_air + thetam[9]

room_mpc = Function('room_mpc', [t_wall, t_room, t_out, power_HP, cop,  thetam], [t_room_mpc])

thetam_num = np.array([1,1,0,1,1,0,1,1,1,0])
################################################################################################
# noise parameters
noise = {   'mu' : { 'room': 0, 'power': 0.033},
            'sig': { 'room': 0.005317236877233841, 'power': 0.2963125110789097},
            'beta': { 'room': 0.99, 'power': 0.92},
            'epsilon': { 'room': 0.73, 'power':  0.68},
}

