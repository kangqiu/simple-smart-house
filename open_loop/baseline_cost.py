from casadi.tools import *
import pytz
import datetime as dt
import numpy as np

#take actuation penalty out!
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
results_file = './results/00_nonoise.pkl'

gamma = 1
episodes = 40

use_ipopt = True


#stepsize of Q-update
# alphal = np.array([0, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3])
alphal = 10
# alphat = 1e-15

################################################################################################
# declaring parametrization
# thetal = MX.sym('thetal', 9)
thetal = MX.sym('thetal', 9)
thetam = MX.sym('theta_m', 9)

thetal_num = np.array([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
# thetal_num = np.array([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0])
# thetal_num = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
# thetal_num = np.array([-1.03716324e-04 , 1.20290061e+00,  3.02950809e+00,  2.23882534e-02,
#   1.00419109e+00,  1.95773548e+00, -3.61380872e-07 ])
thetam_num = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0])


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
t_desired = MX.sym('t_desired')
t_out = MX.sym('t_out')



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

# pow_mpc = thetam[0] * HPset * thetam[1]* t_set - HPin * t_room + HP0 +thetam[2]
k = 0.2
pow =  k * (t_set - t_room)
# new saturation function
alpha = 0.1
powsat = 0.5 * (pow + maxpow - sqrt((pow - maxpow)**2 + alpha))
hppow =  thetam[0] * 0.5 * (powsat + sqrt(powsat ** 2 + alpha)) + thetam[1]
power_mpc = Function('power_mpc', [t_room, t_set, thetam], [hppow])


t_wall_mpc = (1/m_wall) * ( m_wall * t_wall + thetam[2] *rho_out * (t_out - t_wall)
            + thetam[3] * rho_in * (t_room - t_wall) + thetam[4])
wall_mpc = Function('wall_mpc', [t_wall, t_room, t_out, thetam], [t_wall_mpc])

t_room_mpc = (1/m_air) *(m_air * t_room + thetam[5] *rho_in * (t_wall - t_room) + thetam[6] *
            rho_dir * (t_out - t_room) + thetam[7] * cop * power_HP+  thetam[8])


room_mpc = Function('room_mpc', [t_wall, t_room, t_out, power_HP, thetam], [t_room_mpc])

#############################################################################################################
# get cost matching baseline cost
# objective weights
w_spot = 0.075  # weight spot cost
w_tbelow = 0.1 # weight temperature below
w_tabove = 0.05 # weight temperature above
w_tmin = 1.0
w_target = 0.5
hubber = 0.5


# measured stage cost
lmeas = 0
lmeas += w_tabove * (t_desired - t_room_meas) ** 2 / n_rl
lmeas += w_tbelow * fmax((t_room_meas - t_desired), 0) ** 2 / n_rl
lmeas += w_tmin * fmax((t_min - t_room_meas), 0)** 2 / n_rl
lmeas += w_spot * spot * power_HP_meas / n_rl
lmeas_func = Function(
    'lmeas_func', [t_desired, t_room_meas, t_min, power_HP_meas, spot], [lmeas])

# mpc predicted stage cost
# slack variable approximation
b  = 0.001
# squareplus = 0.5*(x+sqrt(x^2+b))
fmax_desired = (t_room - t_desired + thetal[7])
fmax_min = (t_min + thetal[8] - t_room)
# fmax_desired = (t_room - t_desired)
# fmax_min = (t_min + thetal[8] - t_room)

l_mpc = 0
l_mpc += thetal[0]
l_mpc += (thetal[1] * w_tabove * (t_desired- t_room)**2 / float(n_mpc))
l_mpc += (thetal[2] * w_tbelow * 0.5*(fmax_desired + sqrt(fmax_desired **2 +b)) ** 2 / float(n_mpc))
l_mpc += (thetal[3] * w_tmin * 0.5*(fmax_min + sqrt(fmax_min**2+b))**2 / float(n_mpc))
l_mpc += (w_spot * (spot * power_HP) / float(n_mpc))

# l_mpc = 0
# l_mpc += ( w_tabove * (t_desired- t_room)**2 / float(n_mpc))
# l_mpc += (w_tbelow * 0.5*(fmax_desired + sqrt(fmax_desired **2 +b)) ** 2 / float(n_mpc))
# l_mpc += (w_tmin * 0.5*(fmax_min + sqrt(fmax_min**2+b))**2 / float(n_mpc))
# l_mpc += (w_spot * (spot * power_HP) / float(n_mpc))

lmpc_func = Function(
    'lmpc_func', [t_desired, t_min, t_room, power_HP, spot, thetal], [l_mpc])

t_mpc = 0
t_mpc += (thetal[4] * w_tbelow * 0.5 * (fmax_desired +sqrt(fmax_desired ** 2+b)) ** 2 / float(n_mpc))
t_mpc += (thetal[5] * w_tmin * 0.5 * (fmax_min +sqrt(fmax_min **2+b)) ** 2 / float(n_mpc))
t_mpc += thetal[6]

# t_mpc = 0
# t_mpc += ( w_tbelow * 0.5 * (fmax_desired +sqrt(fmax_desired ** 2+b)) ** 2 / float(n_mpc))
# t_mpc += (w_tmin * 0.5 * (fmax_min +sqrt(fmax_min **2+b)) ** 2 / float(n_mpc))



# tmpc_func = Function( 'tmpc_func', [t_room, t_desired, t_min, thetal], [t_mpc])
tmpc_func = Function( 'tmpc_func', [t_room, t_desired, t_min, thetal], [t_mpc])


