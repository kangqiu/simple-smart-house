from casadi.tools import *
import pytz
import datetime as dt
import numpy as np

#take actuation penalty out!
################################################################################################
################################################################################################
# data
n_mpc = 24 * 12 #24h
n_rl = n_mpc
local_timezone = pytz.timezone('Europe/Oslo')

################################################################################################
# declaring parametrization

thetal = MX.sym('thetal', 9)

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

#measured states
t_wall_meas = MX.sym('t_wall_meas')
t_room_meas = MX.sym('t_room_meas')
power_HP_meas = MX.sym('power_HP_meas')

# external parameters
spot = MX.sym('spot')
t_min = MX.sym('t_min')
t_max = MX.sym('t_max')
t_out = MX.sym('t_out')
t_mid = MX.sym('t_mid')



###############################################################################################
# setting up parameterized MPC functions
# parameters
ReLu = 100
maxpow = 1.5
m_air = 31.03
m_wall = 67.22
rho_in = 0.36
rho_out = 0.033
rho_dir = 0.033
cop = 3

#parameterized model equations MPC

k = 0.2
pow =  k * (t_set - t_room)
power_func = Function('power_func', [t_room, t_set], [pow])

# new saturation function

b = 0.01
alpha = 0.5 * (unsatpower + maxpow - sqrt((unsatpower - maxpow)**2 + b))
sat = 0.5 * (alpha + sqrt(alpha ** 2 + b))
satpower_func = Function('satpower_func', [unsatpower], [sat])


wall_plus = (1/m_wall) * ( m_wall * t_wall + rho_out * (t_out - t_wall)
            + rho_in * (t_room - t_wall) )
wall_func = Function('wall_func', [t_wall, t_room, t_out], [wall_plus])

room_plus = (1/m_air) *(m_air * t_room + rho_in * (t_wall - t_room) +
            rho_dir * (t_out - t_room) + cop * power_HP )


room_func = Function('room_func', [t_wall, t_room, t_out, power_HP], [room_plus])

#############################################################################################################
# get cost matching baseline cost
# objective weights
w_spot = 0.8  # weight spot cost
w_max = 1. # weight temperature above
w_min = 50.
w_mid = 1.
w_target = 0.5
hubber = 0.5

# measured stage cost
fmax_max_sim = t_room_meas - t_max
fmax_min_sim = t_min - t_room_meas

lmeas = 0
b = 0.001
lmeas += w_mid * (t_mid - t_room_meas) ** 2 / n_rl
# lmeas += w_max * fmax((t_room_meas - t_max), 0) ** 2 / n_rl
# lmeas += w_min * fmax((t_min - t_room_meas), 0) ** 2 / n_rl
lmeas += w_max * 0.5*(fmax_max_sim + sqrt(fmax_max_sim **2 +b)) ** 2 / n_rl
lmeas += w_min *  0.5*(fmax_min_sim + sqrt(fmax_min_sim**2+b)) ** 2 / n_rl
# lmeas += w_spot * spot * power_HP_meas / n_rl
lmeas_func = Function(
    'lmeas_func', [t_max, t_mid, t_min, t_room_meas, power_HP_meas, spot], [lmeas])

# mpc predicted stage cost
# slack variable approximation
b  = 0.001
slack = (t_room - (t_max + thetal[7]))
slackmin = ((t_min + thetal[8]) - t_room)

lhat = 0
lhat += thetal[0]
lhat += (thetal[1] * w_mid * (t_mid- t_room)**2 / float(n_mpc))
lhat += (thetal[2] * w_max * 0.5*(slack + sqrt(slack **2+b)) ** 2 / float(n_mpc))
lhat += (thetal[3] * w_min * 0.5*(slackmin + sqrt(slackmin **2+b))**2 / float(n_mpc))
# lhat += (w_spot * (spot * power_HP) / float(n_mpc))


lhat_func = Function(
    'lhat_func', [t_max, t_mid, t_min, t_room, power_HP, spot, thetal], [lhat])

that = 0
that += (thetal[4] * w_max * 0.5 * (slack +sqrt(slack ** 2+b)) ** 2 / float(n_mpc))
that += (thetal[5] * w_min * 0.5 * (slackmin +sqrt(slackmin ** 2+b)) ** 2 / float(n_mpc))
that += thetal[6]

that_func = Function( 'that_func', [t_room, t_max,t_mid, t_min, thetal], [that])



#########################################################################################################
### mpc slack functions
slack_mpc = Function('slack_mpc', [t_room, t_max, thetal], [slack])
slackmin_mpc = Function('slackmin_mpc', [t_room, t_min, thetal], [slackmin])

###mpc cost function
slack_sym = MX.sym('slack_sym')
slackmin_sym = MX.sym('slackmin_sym')
dt_target = MX.sym('dt_target')

lmpc = 0
lmpc += thetal[0]
lmpc += (thetal[1] * w_mid * (t_mid- t_room)**2 / float(n_mpc))
lmpc += (thetal[2] * w_max * slack_sym ** 2 / float(n_mpc))
lmpc += (thetal[3] * w_min * slackmin_sym**2 / float(n_mpc))
lmpc += (w_target * (hubber ** 2) * (sqrt(1 + (dt_target / hubber) ** 2) - 1) / float(n_mpc))
lmpc += (w_spot * (spot * power_HP) / float(n_mpc))


lmpc_func = Function(
    'lmpc_func', [t_mid, t_room, slack_sym, slackmin_sym, dt_target, power_HP, spot, thetal], [lmpc])

tmpc = 0
tmpc += (thetal[4] * w_max * slack_sym ** 2 / float(n_mpc))
tmpc += (thetal[5] * w_min * slackmin_sym ** 2 / float(n_mpc))
tmpc += thetal[6]

tmpc_func = Function( 'tmpc_func', [slack_sym, slackmin_sym, thetal], [tmpc])

##########################################
# noise stuff
noise = {'mu': {'room': 0, 'power': 0.033},
             'sig': {'room': 0.00532, 'power': 0.297},
             'beta': {'room': 0.96, 'power': 0.92},
             'epsilon': {'room': 0.75, 'power': 0.68},
             }