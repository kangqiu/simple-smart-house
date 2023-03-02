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
start = dt.datetime(2022, 9, 1, 0, 0).astimezone(local_timezone)
stop = dt.datetime(2022, 10, 1, 0, 0).astimezone(local_timezone)
# outside temperature data
# temp_file = './data/SEKLIMAData_2022.pkl'
set_t_out = 0 # in ˚C
spot_file = './data/SpotData2022_Trheim.pkl'
noise_file = './data/noise/September.pkl'
add_noise = True
# results are saved in this file
results_file = './results/September_tuned_derandomizev2_test.pkl'

# thetal = np.array([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
#complete random trajectory generation -> double check scenario weight though
# thetal = np.array([ 2.61996221e-02,  3.07103216e-01 , 4.93511776e-01,  1.47012626e+00,
#   7.87317446e-01 , 1.00091467e+00 , 9.12878817e-05,  2.49700010e+00,
#  -6.02777968e-01])

# preliminary trained data
thetal =  np.array([ 1.15286429e-03 , 9.37928741e-01 , 6.72211804e-01 , 1.50776098e+00,
  2.37640584e+00 , 1.00000608e+00  ,4.01694873e-06 , 1.14623467e+00,
 -5.19375105e-01])
thetam = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0])

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

k = 0.2
ReLu = 100
maxpow =  1.5
pow = k * (t_target - t_room)
# pow = HPset * t_target - HPin * t_room + HP0

power = Function(
    'power', [t_room, t_target], [pow]
)

unsatpower = MX.sym('unsatpower')
# powsat = log(1 + exp(ReLu * unsatpower)) / ReLu
# hppow = powsat - log(1 + exp(ReLu * (powsat - maxpow))) / ReLu

# new saturation function
alpha = 0.1
powsat = 0.5 * (unsatpower + maxpow - sqrt((unsatpower - maxpow)**2 + alpha))
hppow = 0.5 * (powsat + sqrt(powsat ** 2 + alpha))
satpower = Function('satpower', [unsatpower], [hppow])

#temperature prediction model
m_air = 31.03
m_wall = 67.22
rho_in = 0.36
rho_out = 0.033
rho_dir = 0.033

COP = 3

t_wall_plus = (1/m_wall) * ( m_wall * t_wall
              + rho_out* (t_out - t_wall) + rho_in * (t_room - t_wall))
wallplus = Function('wallplus', [t_wall, t_room, t_out], [t_wall_plus])

power_HP = MX.sym('power_HP')


cop = MX.sym('cop')
t_room_plus = (1/m_air) * (m_air * t_room
             + rho_in * (t_wall - t_room) + rho_dir * (t_out - t_room) + cop * power_HP)

roomplus = Function(
    "roomplus", [t_wall, t_room, t_out, power_HP, cop], [t_room_plus]
)


################################################################################################
# mpc config

solver_options = {
    'print_time': 0,
    'ipopt' : {
    "linear_solver": "ma57",
        'print_level': 0
}}

################################################################################################
# mpc equations
# objective function parameterization
# MPC model parameterization

# MPC objective function
w_spot = 0.8  # weight spot cost
w_max = 5 # weight temperature above
w_min = 10.
w_mid = 1
w_target = 0.5
hubber = 0.5

# define symbolic variables
tmax = MX.sym('tmax')
tmin = MX.sym('tmin')
tmid = MX.sym('tmid')

dt_target = MX.sym('dt_target')
slack = MX.sym('slack')
slackmin = MX.sym('slackmin')
spot = MX.sym('spot')


#define symbolic parameterized stage cost function
l_mpc = 0
l_mpc += thetal[0]
l_mpc += (thetal[1] * w_mid * (tmid - t_room)**2 / float(n_mpc))

l_mpc += (thetal[2] * w_max * slack ** 2 / float(n_mpc))
l_mpc += (thetal[3] * w_min * slackmin ** 2 / float(n_mpc))
l_mpc += (w_target * (hubber ** 2) * (sqrt(1 + (dt_target / hubber) ** 2) - 1) / float(n_mpc))
l_mpc += (w_spot * (spot * power_HP) / float(n_mpc))

lmpc_func = Function(
    'lmpc_func', [tmid, t_room, slack, slackmin, dt_target, power_HP, spot], [l_mpc]
)

#parameterized terminal cost function

t_mpc = 0
t_mpc += (thetal[4] * w_max * slack ** 2 / float(n_mpc))
t_mpc += (thetal[5] * w_min * slackmin ** 2 / float(n_mpc))
t_mpc += thetal[6]

tmpc_func = Function( 'tmpc_func', [slack, slackmin], [t_mpc])

#parameterized model equations MPC

pow =  k * (t_target - t_room)
powsat = 0.5 * (pow + maxpow - sqrt((pow - maxpow)**2 + alpha))
hppow = thetam[0] * 0.5 * (powsat + sqrt(powsat ** 2 + alpha)) + thetam[1]
power_mpc = Function('power_mpc', [t_room, t_target], [hppow])

t_wall_mpc = (1/m_wall) * ( m_wall * t_wall + thetam[2] *rho_out * (t_out - t_wall)
            + thetam[3] * rho_in * (t_room - t_wall) + thetam[4])
wall_mpc = Function('wall_mpc', [t_wall, t_room, t_out], [t_wall_mpc])

t_room_mpc = (1/m_air) *(m_air * t_room + thetam[5] *rho_in * (t_wall - t_room) + thetam[6] * 
                          rho_dir * (t_out - t_room) + thetam[7] * cop * power_HP+  thetam[8])
    
room_mpc = Function('room_mpc', [t_wall, t_room, t_out, power_HP, cop], [t_room_mpc])

################################################################################################
# noise parameters
# noise = {   'mu' : { 'room': 0, 'power': 0.033},
#             'sig': { 'room': 0.005317236877233841, 'power': 0.2963125110789097},
#             'beta': { 'room': 0.99, 'power': 0.92},
#             'epsilon': { 'room': 0.73, 'power':  0.68},
# }
#bigger room noise
noise = {   'mu' : { 'room': 0, 'power': 0.033},
            'sig': { 'room': 0.00532, 'power': 0.297},
            'beta': { 'room': 0.96, 'power': 0.92},
            'epsilon': { 'room': 0.75, 'power':  0.68},
}
