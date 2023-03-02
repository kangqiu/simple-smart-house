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
# new saturation function
alpha = 0.1
powsat = 0.5 * (pow + maxpow - sqrt((pow - maxpow)**2 + alpha))
hppow =  0.5 * (powsat + sqrt(powsat ** 2 + alpha))
power_mpc = Function('power_mpc', [t_room, t_set], [hppow])


t_wall_mpc = (1/m_wall) * ( m_wall * t_wall + rho_out * (t_out - t_wall)
            + rho_in * (t_room - t_wall) )
wall_mpc = Function('wall_mpc', [t_wall, t_room, t_out], [t_wall_mpc])

t_room_mpc = (1/m_air) *(m_air * t_room + rho_in * (t_wall - t_room) +
            rho_dir * (t_out - t_room) + cop * power_HP )


room_mpc = Function('room_mpc', [t_wall, t_room, t_out, power_HP], [t_room_mpc])

#############################################################################################################
# get cost matching baseline cost
# objective weights
w_spot = 0.8  # weight spot cost
w_max = 5 # weight temperature above
w_min = 10.
w_mid = 1

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
# squareplus = 0.5*(x+sqrt(x^2+b))
fmax_max= (t_room - t_max + thetal[7])
fmax_min = (t_min + thetal[8] - t_room)
# fmax_desired = (t_room - t_desired)
# fmax_min = (t_min + thetal[8] - t_room)

l_mpc = 0
l_mpc += thetal[0]
l_mpc += (thetal[1] * w_mid * (t_mid- t_room)**2 / float(n_mpc))
l_mpc += (thetal[2] * w_max * 0.5*(fmax_max + sqrt(fmax_max **2 +b)) ** 2 / float(n_mpc))
l_mpc += (thetal[3] * w_min * 0.5*(fmax_min + sqrt(fmax_min**2+b))**2 / float(n_mpc))
# l_mpc += (w_spot * (spot * power_HP) / float(n_mpc))

# l_mpc = 0
# l_mpc += ( w_tabove * (t_desired- t_room)**2 / float(n_mpc))
# l_mpc += (w_tbelow * 0.5*(fmax_desired + sqrt(fmax_desired **2 +b)) ** 2 / float(n_mpc))
# l_mpc += (w_tmin * 0.5*(fmax_min + sqrt(fmax_min**2+b))**2 / float(n_mpc))
# l_mpc += (w_spot * (spot * power_HP) / float(n_mpc))

lmpc_func = Function(
    'lmpc_func', [t_max, t_mid, t_min, t_room, power_HP, spot, thetal], [l_mpc])

t_mpc = 0
t_mpc += (thetal[4] * w_max * 0.5 * (fmax_max +sqrt(fmax_max** 2+b)) ** 2 / float(n_mpc))
t_mpc += (thetal[5] * w_min * 0.5 * (fmax_min +sqrt(fmax_min **2+b)) ** 2 / float(n_mpc))
t_mpc += thetal[6]

# t_mpc = 0
# t_mpc += ( w_tbelow * 0.5 * (fmax_desired +sqrt(fmax_desired ** 2+b)) ** 2 / float(n_mpc))
# t_mpc += (w_tmin * 0.5 * (fmax_min +sqrt(fmax_min **2+b)) ** 2 / float(n_mpc))



# tmpc_func = Function( 'tmpc_func', [t_room, t_desired, t_min, thetal], [t_mpc])
tmpc_func = Function( 'tmpc_func', [t_room, t_max,t_mid, t_min, thetal], [t_mpc])



