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

    # ax2.set_xlabel('time')
    # ax2.set_ylabel('power consumption [kW]', color='green')
    # ax2.plot(
    #     timesteps, df_history['power'].values.tolist(), label="Power")
    # ax2.plot(
    #     timesteps, power_trajectory, label="power_mpc")
    # ax2.tick_params(axis="x", labelrotation=45)
    # ax3 = ax2.twinx()
    # ax3.set_ylabel("spot pricing", color='green')
    # ax3.plot(
    #     timesteps, df_history['spot'].values.tolist(),
    #     label="Spot",
    #     color='orange'
    # )
    # ax3.grid()

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

#test data
results_file = './results/derandomizev2_run1.pkl'
history_runs = read_results(results_file)

# tuned thetal
thetal = np.array([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])

thetal_tuned = np.array([ 1.15286429e-03 , 9.37928741e-01 , 6.72211804e-01 , 1.50776098e+00,
  2.37640584e+00 , 1.00000608e+00  ,4.01694873e-06 , 1.14623467e+00,
 -5.19375105e-01])
#

#build symbolic casadi function to evaluate predicted Q by one (1) MPC iteration
Qmpc = get_Qmpc()

Qbase = get_Qbase()
# to test: Qbase(t_desired_num[0:cfg.n_rl], room_num[0:cfg.n_rl], t_min_num[0:cfg.n_rl], dt_set_num[0:cfg.n_rl], power_num[0:cfg.n_rl], spot_num[0:cfg.n_rl])
# batches = create_batches(df_history)

w = MX.sym('w', 9)
J = 0
Q_mpc_tot = []
Q_mpc_theta_tot = []
Q_base_tot = []
residuals = []
residuals_theta = []
for k, index in enumerate(tqdm(range(len(history_runs)))):
    df_history = history_runs[k]
    t_set_num, t_out_num, room_num, power_num, wall_num, spot_num, tmin_num, tmid_num, tmax_num = get_numerical_data(df_history)
    ts = 0
    Q_mpc_tot.append(Qmpc(room_num[ts], wall_num[ts],
         t_set_num[ts:ts + form.n_mpc], tmax_num[ts:ts + form.n_mpc], tmid_num[ts:ts + form.n_mpc],
         tmin_num[ts:ts + form.n_mpc],
         t_out_num[ts:ts + form.n_mpc], spot_num[ts:ts + form.n_mpc], thetal).full().flatten()[0])

    Q_mpc_theta_tot.append(Qmpc(room_num[ts], wall_num[ts],
                          t_set_num[ts:ts + form.n_mpc], tmax_num[ts:ts + form.n_mpc], tmid_num[ts:ts + form.n_mpc],
                          tmin_num[ts:ts + form.n_mpc],
                          t_out_num[ts:ts + form.n_mpc], spot_num[ts:ts + form.n_mpc], thetal_tuned).full().flatten()[0])

    Q_base_tot.append(Qbase(tmax_num[ts:ts + form.n_rl], tmid_num[ts:ts + form.n_rl],
          tmin_num[ts:ts + form.n_rl], room_num[ts:ts + form.n_rl],
          power_num[ts:ts + form.n_rl],
          spot_num[ts:ts + form.n_rl]).full().flatten()[0])

    residuals.append(Q_mpc_tot[-1] - Q_base_tot[-1])
    residuals_theta.append(Q_mpc_theta_tot[-1] - Q_base_tot[-1])



bins = np.linspace(min(min(residuals), min(residuals_theta)), max(max(residuals), max(residuals_theta)), 100)
plt.hist(residuals, bins, alpha=0.5, label='residuals')
plt.hist(residuals_theta, bins, alpha=0.5, label='residuals tuned')
plt.legend(loc='upper right')
plt.grid()
plt.show()

print('untuned residuals', sum(residuals))
print('tuned residuals', sum(residuals_theta))
print('residuals postive meaning mpc overestimates')