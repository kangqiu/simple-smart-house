from datetime import date, datetime, timedelta
import pytz
import matplotlib.pyplot as plt
import pandas as pd
from casadi.tools import *
from tqdm import tqdm
from time import sleep
import pickle as pkl
import formulation as form
import scipy.interpolate as scinterp

def plot_state_trajectories(k, history_runs):
    # plot state trajectories
    df_history = history_runs[k]
    t_set_num, t_out_num, room_num, power_num, wall_num, spot_num, tmin_num, tmid_num, tmax_num = get_numerical_data(
        df_history)
    ts = 0
    room_trajectory, wall_trajectory, power_trajectory = get_mpc_state_trajectories(room_num[ts], wall_num[ts],
                                                                                    t_set_num[ts:ts + form.n_mpc],
                                                                                    t_out_num[ts:ts + form.n_mpc])

    # plot mpc trajectories
    timesteps = range(len(df_history.index))
    fig, (ax1) = plt.subplots(1)
    ax1.plot(timesteps, df_history['tmax'].values.tolist(), label='tmax')
    ax1.plot(timesteps, df_history['tmid'].values.tolist(), label='tmid')
    ax1.plot(timesteps, df_history['tmin'].values.tolist(), label='tmin')
    ax1.plot(timesteps, df_history['room'].values.tolist(), label='room measured')
    ax1.plot(timesteps, room_trajectory, label='room mpc')
    ax1.set_xticklabels([])
    ax1.tick_params(axis="x",
                    labelrotation=45,  # changes apply to the x-axis
                    which="both",  # both major and minor ticks are affected
                    bottom=False,  # ticks along the bottom edge are off
                    top=False,
                    )
    ax1.grid()
    handles, labels = ax1.get_legend_handles_labels()

    fig.legend(handles, labels)  # loc='upper right')
    plt.tight_layout()

    plt.grid("on")

    plt.show()

def read_results(results_file):
    with open(results_file, 'rb') as handle:
        history = pkl.load(handle)
    return history

def get_numerical_data(df_history):
    room_num = df_history['room'].values
    wall_num = df_history['wall'].values
    power_num = df_history['power'].values
    # get action
    t_set_num = df_history['target'].values
    # external params
    t_out_num = df_history['tout'].values
    spot_num = df_history['spot'].values

    #settings
    tmin = df_history['tmin'].values
    tmid = df_history['tmid'].values
    tmax = df_history['tmax'].values
    return t_set_num, t_out_num, room_num, power_num, wall_num, spot_num, tmin, tmid, tmax

def get_mpc_state_trajectories(room0, wall0, tset, tout):
    room = room0
    wall = wall0
    room_trajectory = [room0]
    wall_trajectory = [wall0]
    power_trajectory = []

    for k in range(form.n_mpc - 1):
        power = form.power_mpc(room, tset[k]).full().flatten()[0]
        power_trajectory.append(power)
        # update state variables for prediction
        wall_pred = form.wall_mpc(wall, room, tout[k]).full().flatten()[0]
        room_pred = form.room_mpc(wall, room, tout[k], power).full().flatten()[0]
        wall_trajectory.append(wall_pred)
        room_trajectory.append(room_pred)

        wall = wall_pred
        room = room_pred
    power = form.power_mpc(room, tset[-1]).full().flatten()[0]
    power_trajectory.append(power)

    return room_trajectory, wall_trajectory, power_trajectory


def get_Qmpc():
    # measured states
    room0 = MX.sym('room0')
    wall0 = MX.sym('wall0')
    # initialize state variables
    room = room0
    wall = wall0

    tset = MX.sym('tset', form.n_mpc)
    dtset = MX.sym('dtset', form.n_mpc)
    tout = MX.sym('tout', form.n_mpc)
    tmax= MX.sym('tdesired', form.n_mpc)
    tmid = MX.sym('tmid', form.n_mpc)
    tmin = MX.sym('tmin', form.n_mpc)
    spot = MX.sym('spot', form.n_mpc)

    Qmpc = 0
    for k in range(form.n_mpc-1):
        power = form.power_mpc(room, tset[k+1])
        l_mpc = form.lmpc_func(tmax[k], tmid[k], tmin[k], room, power, spot[k], form.thetal)
        Qmpc += l_mpc

        # update state variables for prediction
        wall_pred = form.wall_mpc(wall, room, tout[k])
        room_pred = form.room_mpc(wall, room, tout[k], power)

        wall = wall_pred
        room = room_pred

    t_mpc = form.tmpc_func(room[-1], tmax[-1], tmid[-1], tmin[-1], form.thetal)
    Qmpc += t_mpc
    Qmpc_func = Function('Qmpc_func',
                         [room0, wall0, tset, tmax, tmid, tmin, tout, spot, form.thetal],
                         [Qmpc])
    return Qmpc_func

def get_Qbase():
    # measured states
    roommeas = MX.sym('roommeas', form.n_rl)
    wallmeas = MX.sym('wallmeas', form.n_rl)
    powermeas = MX.sym('powermeas', form.n_rl)
    tset = MX.sym('tset', form.n_rl)
    dtset = MX.sym('dtset', form.n_rl)
    tout = MX.sym('tout', form.n_rl)
    tmax = MX.sym('tmax', form.n_rl)
    tmid = MX.sym('tmid', form.n_rl)
    tmin = MX.sym('tmin', form.n_rl)
    spot = MX.sym('spot', form.n_rl)

    Qbase = 0
    for k in range(form.n_rl):
        l_meas = form.lmeas_func(tmax[k], tmin[k], tmin[k], roommeas[k], powermeas[k], spot[k])
        Qbase += l_meas

    Qbase_func = Function('Qbase_func',
                         (tmax, tmid, tmin, roommeas, powermeas, spot),
                         [Qbase])

    return Qbase_func


###############################################################################################
# decide which spot market to read
time_start = datetime(2022, 9, 1, 0, 0).astimezone(form.local_timezone)
spot_file = '../data/SpotData2022_Trheim.pkl'
results_file = './results/derandomizev2_run1_test.pkl'
print(results_file)
history_runs = read_results(results_file)

thetal = np.array([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
ubw = [+inf] * 9
lbw = [-inf] + [0] * 5 + [-inf]*3

#build symbolic casadi function to evaluate predicted Q by one (1) MPC iteration
Qmpc = get_Qmpc()

Qbase = get_Qbase()

w = MX.sym('w', 9)
J = 0
Q_mpc_tot = []
Q_base_tot = []
residuals = []
for k, index in enumerate(tqdm(range(len(history_runs)))):
    df_history = history_runs[k]
    t_set_num, t_out_num, room_num, power_num, wall_num, spot_num, tmin_num, tmid_num, tmax_num = get_numerical_data(df_history)
    ts = 0
    Q_mpc_tot.append(Qmpc(room_num[ts], wall_num[ts],
         t_set_num[ts:ts + form.n_mpc], tmax_num[ts:ts + form.n_mpc], tmid_num[ts:ts + form.n_mpc],
         tmin_num[ts:ts + form.n_mpc],
         t_out_num[ts:ts + form.n_mpc], spot_num[ts:ts + form.n_mpc], thetal).full().flatten()[0])

    Q_base_tot.append(Qbase(tmax_num[ts:ts + form.n_rl], tmid_num[ts:ts + form.n_rl],
          tmin_num[ts:ts + form.n_rl], room_num[ts:ts + form.n_rl],
          power_num[ts:ts + form.n_rl],
          spot_num[ts:ts + form.n_rl]).full().flatten()[0])

    residuals.append(Q_mpc_tot[-1] - Q_base_tot[-1])

    res = (Qmpc(room_num[ts], wall_num[ts],
         t_set_num[ts:ts + form.n_mpc], tmax_num[ts:ts + form.n_mpc], tmid_num[ts:ts + form.n_mpc],
         tmin_num[ts:ts + form.n_mpc],
         t_out_num[ts:ts + form.n_mpc], spot_num[ts:ts + form.n_mpc], w)
          - Qbase(tmax_num[ts:ts + form.n_rl], tmid_num[ts:ts + form.n_rl],
          tmin_num[ts:ts + form.n_rl], room_num[ts:ts + form.n_rl],
          power_num[ts:ts + form.n_rl],
          spot_num[ts:ts + form.n_rl]))

    J += 0.5 * (res) ** 2 / len(history_runs)

# #add regularization to slack variable tuning
for t in range(len(thetal)):
    J += 1e-2 * (w[t] - thetal[t])**2
#

# plot base cost and mpc cost predicted
plt.plot(range(len(Q_base_tot)), Q_base_tot, label = 'base')
plt.plot(range(len(Q_mpc_tot)), Q_mpc_tot, label = 'mpc')
plt.legend()
plt.grid()
plt.show()


bins = np.linspace(min(residuals), max(residuals), 100)
plt.hist(residuals, bins, alpha=0.5, label='residuals')
plt.legend(loc='upper right')
plt.show()

print(sum(residuals))


g = []
lbg = []
ubg = []
#
LS = {'f': J, 'x': w, 'g': g, 'p': []}
options = {'print_time': 1}
options['ipopt'] = {'linear_solver': 'ma57',
                    'max_iter': 50,
                    'print_level': 5
                    }
solverLS = nlpsol('solver', 'ipopt', LS, options)
# w0 = np.concatenate((cfg.thetal_num, cfg.thetam_num), axis=None)
w0 = thetal

sol = solverLS(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=[])
w_opt = sol['x'].full().flatten()

print(w_opt)


#get max residual
k = residuals.index(max(residuals))
plot_state_trajectories(k, history_runs)


#get min residual
k = residuals.index(min(residuals))
plot_state_trajectories(k, history_runs)

