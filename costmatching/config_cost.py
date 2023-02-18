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
n_mpc = 24 * 12 #24h
n_rl = 24 * 12 #24h
n_batches = 31 #days
len_batches = 288*5 #2 days length batchsize
#results are saved in this file
results_file = '../results/newsat/01_training_january.pkl'

gamma = 1
episodes = 3

#stepsize of Q-update
alphal = np.array([0, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3])
alpham = 1e-15
alphat = 1e-15

constrained_update = False

use_ipopt = True

################################################################################################
# declaring parametrization
thetal = MX.sym('thetal', 9)
thetam = MX.sym('theta_m', 9)

thetal_num = np.array([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0])
thetam_num = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0])

# setting lower bound for parameter updates
# thetal_lb = np.array([0, 0.001, 0.001, 0.001])


################################################################################################
# setting up symbolic casadi variables
# action
t_set = MX.sym('t_set')
dt_set = MX.sym('dt_set')

#states
t_wall = MX.sym('t_wall')
t_room = MX.sym('t_room')
unsatpower = MX.sym('unsatpower')
power_HP = MX.sym('power_HP')
slack = MX.sym('slack')
slackmin = MX.sym('slackmin')

#measured states
t_wall_meas = MX.sym('t_wall_meas')
t_room_meas = MX.sym('t_room_meas')
power_HP_meas = MX.sym('power_HP_meas')

# external parameters
spot = MX.sym('spot')
t_min = MX.sym('t_min')
t_desired = MX.sym('t_desired')
t_out = MX.sym('t_out')



###############################################################################################
# setting up parameterized MPC functions
# parameters
HP0 = 1.1441609877948857
HPin = 0.08436238305547132
HPset = 0.06071641164806341
ReLu = 100
maxpow = 1.5
m_air = 31.02679204362912
m_wall = 67.21826736655125
rho_in = 0.36409940390361406
rho_out = 0.03348756113438382
rho_dir = 0.03348756179891388
COP = 3
rho_out_wall = rho_out/m_wall
rho_in_wall = rho_in/m_wall
rho_in_room = rho_out/m_air
rho_dir_room = rho_dir/m_air

#parameterized model equations MPC

# pow_mpc = thetam[0] * HPset * thetam[1]* t_set - HPin * t_room + HP0 +thetam[2]
k = 0.2
pow_mpc = thetam[0] * k * (t_set - t_room) + thetam[1]

power_mpc = Function('power_mpc', [t_room, t_set, thetam], [pow_mpc])

alpha = 0.1
powsat = 0.5 * (unsatpower + maxpow - sqrt((unsatpower - maxpow)**2 + alpha))
hppow = 0.5 * (powsat + sqrt(powsat ** 2 + alpha))
satpower_mpc  = Function('satpower_mpc', [unsatpower], [hppow])

# powsat = log(1 + exp(ReLu * unsatpower)) / ReLu
# hppow = powsat - log(1 + exp(ReLu * (powsat - maxpow))) / ReLu
# satpower_mpc = Function('satpower_mpc', [unsatpower], [hppow])


t_wall_mpc = t_wall + thetam[2] * rho_out_wall * (t_out - t_wall) + thetam[3] * rho_in_wall * (t_room - t_wall) + thetam[4]
wall_mpc = Function('wall_mpc', [t_wall, t_room, t_out, thetam], [t_wall_mpc])

t_room_mpc = t_room + thetam[5]*rho_in_room * (t_wall - t_room) + thetam[6] * rho_dir_room * (t_out - t_room) + thetam[7] *\
             COP * power_HP * 1/m_air + thetam[8]

room_mpc = Function('room_mpc', [t_wall, t_room, t_out, power_HP, thetam], [t_room_mpc])

#############################################################################################################
# get cost matching baseline cost
# objective weights
w_spot = 0.1
w_tbelow = 0.2
w_tabove = 0.005
w_tmin = 50
w_tset = 0.5
hubber = 0.5

# measured stage cost
lmeas = 0
lmeas += w_tabove * (t_desired - t_room_meas) ** 2 / n_rl
lmeas += w_tbelow * fmax((t_room_meas - t_desired), 0) ** 2 / n_rl
lmeas += w_tmin * fmax((t_min - t_room_meas), 0)** 2 / n_rl
lmeas += w_tset * (hubber ** 2) * (sqrt(1 + (dt_set/ hubber) ** 2) - 1) / n_rl
lmeas += w_spot * spot * power_HP_meas / n_rl
lmeas_func = Function(
    'lmeas_func', [t_desired, t_room_meas, t_min, dt_set, power_HP_meas, spot], [lmeas])

# mpc predicted stage cost
l_mpc = 0
l_mpc += thetal[0]
l_mpc += (thetal[1] * w_tabove * (t_desired- t_room)**2 / float(n_mpc))
l_mpc += (thetal[2] * w_tbelow * fmax((t_room - t_desired), 0) ** 2 / float(n_mpc))
l_mpc += (thetal[3] * w_tmin * fmax((t_min - t_room),0) ** 2 / float(n_mpc))
l_mpc += (thetal[4] * w_tset * (hubber ** 2) * (sqrt(1 + (dt_set / hubber) ** 2) - 1) / float(n_mpc))
l_mpc += (thetal[5] * w_spot * (spot * power_HP) / float(n_mpc))

lmpc_func = Function(
    'lmpc_func', [t_desired, t_min, t_room, dt_set, power_HP, spot, thetal], [l_mpc])

t_mpc = 0
t_mpc += (thetal[6] * w_tbelow * fmax((t_room - t_desired), 0) ** 2 / float(n_mpc))
t_mpc += (thetal[7] * w_tmin * fmax((t_min - t_room),0) ** 2 / float(n_mpc))
t_mpc += thetal[8]

tmpc_func = Function( 'tmpc_func', [t_room, t_desired, t_min, thetal], [t_mpc])

