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
        power = form.power_func(room, tset[k]).full().flatten()[0]
        power = form.satpower_func(power).full().flatten()[0]
        power_trajectory.append(power)
        # update state variables for prediction
        wall_pred = form.wall_func(wall, room, tout[k]).full().flatten()[0]
        room_pred = form.room_func(wall, room, tout[k], power).full().flatten()[0]
        wall_trajectory.append(wall_pred)
        room_trajectory.append(room_pred)

        wall = wall_pred
        room = room_pred
    power = form.power_func(room, tset[k]).full().flatten()[0]
    power = form.satpower_func(power).full().flatten()[0]
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
        power = form.power_func(room, tset[k])
        power = form.satpower_func(power)
        l_mpc = form.lhat_func(tmax[k], tmid[k], tmin[k], room, power, spot[k], form.thetal)
        Qmpc += l_mpc

        # update state variables for prediction
        wall_pred = form.wall_func(wall, room, tout[k])
        room_pred = form.room_func(wall, room, tout[k], power)

        wall = wall_pred
        room = room_pred

    t_mpc = form.that_func(room[-1], tmax[-1], tmid[-1], tmin[-1], form.thetal)
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
        l_meas = form.lmeas_func(tmax[k], tmid[k], tmin[k], roommeas[k], powermeas[k], spot[k])
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
results_file = './results/02_openloop_week.pkl'
history_runs = read_results(results_file)

# tuned thetal
thetal = np.array([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])

thetal_tuned = np.array([6.17219637e-04, 1.11016402e+00, 9.99789794e-01, 9.75831310e-01,
 9.84418325e-01, 1.00000008e+00, 2.15059108e-06, 1.15412204e-01,
 7.41456376e-02]
)
# thetal = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
#
# thetal_tuned = np.array([ 0.47933593,  0.63925565,  1.05999054,  1.06281101,  1.00000001,  1.31959097,
#  -0.15161482])

#build symbolic casadi function to evaluate predicted Q by one (1) MPC iteration
Qmpc = get_Qmpc()

Qbase = get_Qbase()
# to test: Qbase(t_desired_num[0:cfg.n_rl], room_num[0:cfg.n_rl], t_min_num[0:cfg.n_rl], dt_set_num[0:cfg.n_rl], power_num[0:cfg.n_rl], spot_num[0:cfg.n_rl])
# batches = create_batches(df_history)

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



#get max residual
k = residuals.index(max(residuals))
plot_state_trajectories(k, history_runs)

#get min residual
k = residuals.index(min(residuals))
plot_state_trajectories(k, history_runs)

bins = np.linspace(min(min(residuals), min(residuals_theta)), max(max(residuals), max(residuals_theta)), 100)
plt.hist(residuals, bins, alpha=0.5, label='residuals')
plt.hist(residuals_theta, bins, alpha=0.5, label='residuals tuned')
plt.legend(loc='upper right')
plt.grid()
plt.show()

print('untuned residuals', sum(residuals))
print('tuned residuals', sum(residuals_theta))

#plot comfort cost
y = []
x = np.linspace(15, 25, 1000)

for i in x:
    y_int = form.w_mid * (i - 20) ** 2
    if i < 17:
        y_int += form.w_min * (17 - i) ** 2
    if i > 22 :
        y_int += form.w_max * (i - 22) ** 2
    y.append(y_int)

y_tuned = []
for i in x:
    y_int = thetal_tuned[1] * form.w_mid * (i - 20) ** 2
    if i < 17 + thetal_tuned[8]:
        y_int += thetal_tuned[2] * form.w_min * (17 + thetal_tuned[8] - i) ** 2
    if i > 22 -thetal_tuned[7]:
        y_int += thetal_tuned[3] * form.w_max  * (i - 22 + thetal_tuned[7]) ** 2
    y_tuned.append(y_int)

plt.plot(x, y, label='base')
# plt.axvline(x = 20, color = 'blue')
plt.axvline(x = 17, label = 'min', color = 'r')
plt.axvline(x = 22, label = 'max', color = 'r')
plt.plot(x, y_tuned, label='tuned')
# plt.axvline(x = 20, color = 'orange')
plt.axvline(x = 17+ thetal_tuned[8],  label = 'min tuned', color = 'green')
plt.axvline(x = 22 + thetal_tuned[7], label = 'max tuned',  color = 'green')
plt.legend()
plt.grid()
plt.show()
