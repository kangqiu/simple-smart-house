from datetime import date, datetime, timedelta
import pytz
import matplotlib.pyplot as plt
import pandas as pd
from casadi.tools import *
from tqdm import tqdm
from time import sleep
import datahandling
import config_cost as cfg
import pickle as pkl

def read_results(file):
    with open(file, 'rb') as handle:
        history = pkl.load(handle)
    return history

def plot_open_loop_at_timestep(ts, df_history, thetam):
    # get mpc prediction of state trajectory
    t_set_num, dt_set_num, t_out_num, t_desired_num, t_min_num, spot_num, \
    room_num, power_num, wall_num, room_noise, power_noise = get_numerical_data(df_history)
    power_mpc = []
    room_mpc = [room_num[ts]]
    wall_mpc = [wall_num[ts]]
    for k in range(cfg.n_mpc - 1):
        power_step = cfg.power_mpc(room_mpc[-1], t_set_num[ts + 1 + k], thetam).full().flatten()[0]
        power_mpc.append(power_step)
        # update state variables for prediction
        wall_pred = cfg.wall_mpc(wall_mpc[-1], room_mpc[-1], t_out_num[ts + k], thetam).full().flatten()[0]
        room_pred = \
        cfg.room_mpc(wall_mpc[-1], room_mpc[-1], t_out_num[ts + k], power_step, thetam).full().flatten()[0]

        wall_mpc.append(wall_pred)
        room_mpc.append(room_pred)

    power_step = cfg.power_mpc(room_mpc[-1], t_set_num[ts + 1 + cfg.n_mpc], thetam).full().flatten()[0]
    power_mpc.append(power_step)

    # plot open and closed loop trajectories
    plt.subplot(3, 1, 1)
    plt.plot(range(cfg.n_mpc), room_mpc, label='room mpc')
    plt.plot(range(cfg.n_mpc), room_num[ts:ts + cfg.n_mpc], label='room base')
    plt.plot(range(cfg.n_mpc), wall_mpc, label='wall mpc')
    plt.plot(range(cfg.n_mpc), wall_num[ts:ts + cfg.n_mpc], label='wall base')
    plt.grid('on')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(range(cfg.n_mpc), power_mpc, label='power mpc')
    plt.plot(range(cfg.n_mpc), power_num[ts:ts + cfg.n_mpc], label='power base')
    plt.grid('on')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(range(cfg.n_mpc), room_noise[ts:ts + cfg.n_mpc], label='room noise')
    plt.grid('on')
    plt.legend()
    plt.show()

def get_numerical_data(df_history):
    room_num = df_history['room'].values
    wall_num = df_history['wall'].values
    power_num = df_history['power'].values
    # get action
    t_set_num = df_history['target'].values
    dt_set_num = datahandling.get_dtset(t_set_num)
    # external params
    t_out_num = df_history['t_out'].values
    t_desired_num = df_history['t_desired'].values
    t_min_num = df_history['t_min'].values
    spot_num = df_history['spot_price'].values
    room_noise = df_history['room_noise'].values
    power_noise = np.array([i * 0.3 for i in df_history['power_noise'].values])
    return (t_set_num, dt_set_num, t_out_num, t_desired_num,
            t_min_num, spot_num, room_num, power_num, wall_num,
            room_noise, power_noise)

def get_Qmpc():
    # measured states
    room0 = MX.sym('room0')
    wall0 = MX.sym('wall0')
    # initialize state variables
    room = room0
    wall = wall0

    tset = MX.sym('tset', cfg.n_mpc)
    dtset = MX.sym('dtset', cfg.n_mpc)
    tout = MX.sym('tout', cfg.n_mpc)
    tdesired = MX.sym('tdesired', cfg.n_mpc)
    tmin = MX.sym('tmin', cfg.n_mpc)
    spot = MX.sym('spot', cfg.n_mpc)

    Qmpc = 0
    for k in range(cfg.n_mpc-1):
        # power_unsat = cfg.power_mpc(room, tset[k], cfg.thetam)
        power = cfg.power_mpc(room, tset[k+1], cfg.thetam_num)

        l_mpc = cfg.lmpc_func(tdesired[k], tmin[k], room, power, spot[k], cfg.thetal)
        Qmpc += l_mpc

        # update state variables for prediction
        wall_pred = cfg.wall_mpc(wall, room, tout[k], cfg.thetam)
        room_pred = cfg.room_mpc(wall, room, tout[k], power, cfg.thetam)

        wall = wall_pred
        room = room_pred

    t_mpc = cfg.tmpc_func(room[-1], tdesired[-1], tmin[-1], cfg.thetal)
    Qmpc += t_mpc
    Qmpc_func = Function('Qmpc_func',
                         [room0, wall0, tset, tdesired, tmin, tout, spot, cfg.thetal, cfg.thetam],[Qmpc])

    return Qmpc_func #, dQmpc_l, dQmpc_m

def get_Qbase():
    # measured states
    roommeas = MX.sym('roommeas', cfg.n_rl)
    wallmeas = MX.sym('wallmeas', cfg.n_rl)
    powermeas = MX.sym('powermeas', cfg.n_rl)
    tset = MX.sym('tset', cfg.n_rl)
    tout = MX.sym('tout', cfg.n_rl)
    tdesired = MX.sym('tdesired', cfg.n_rl)
    tmin = MX.sym('tmin', cfg.n_rl)
    spot = MX.sym('spot', cfg.n_rl)

    Qbase = 0
    for k in range(cfg.n_rl):
        l_meas = cfg.lmeas_func(tdesired[k], roommeas[k], tmin[k], powermeas[k], spot[k])
        Qbase += l_meas

    # Qbase += (cfg.w_tbelow * fmax((roommeas[cfg.n_rl-1] - tdesired[cfg.n_rl-1]), 0) ** 2 / float(cfg.n_rl))
    # Qbase += (cfg.w_tmin * fmax((tmin[cfg.n_rl-1] - roommeas[cfg.n_rl-1 ]),0) ** 2 / float(cfg.n_rl))

    Qbase_func = Function('Qbase_func',
                         (tdesired, roommeas, tmin,  powermeas, spot),
                         [Qbase])

    return Qbase_func

def get_catergorized_Qmpc():
    # measured states
    room0 = MX.sym('room0')
    wall0 = MX.sym('wall0')
    # initialize state variables
    room = room0
    wall = wall0

    tset = MX.sym('tset', cfg.n_mpc)
    tout = MX.sym('tout', cfg.n_mpc)
    tdesired = MX.sym('tdesired', cfg.n_mpc)
    tmin = MX.sym('tmin', cfg.n_mpc)
    spot = MX.sym('spot', cfg.n_mpc)

    Q_comfort = 0
    Q_spot = 0
    for k in range(cfg.n_mpc - 1):
        power = cfg.power_mpc(room, tset[k + 1], cfg.thetam_num)

        l_comfort = (cfg.thetal[1] * cfg.w_tabove * (tdesired[k] - room) ** 2 / float(cfg.n_mpc))
        # l_comfort += (cfg.thetal[2] * cfg.w_tbelow * fmax((room + cfg.thetal[3] - tdesired[k]), 0) ** 2 / float(cfg.n_mpc))
        # l_comfort += (cfg.thetal[4] * cfg.w_tmin * fmax((tmin[k] + cfg.thetal[5] - room), 0) ** 2 / float(cfg.n_mpc))
        l_comfort += ( cfg.w_tbelow * fmax((room + cfg.thetal[2] - tdesired[k]), 0) ** 2 / float(
            cfg.n_mpc))
        l_comfort += (cfg.w_tmin * fmax((tmin[k] + cfg.thetal[3] - room), 0) ** 2 / float(cfg.n_mpc))

        Q_comfort += l_comfort

        l_spot = (cfg.w_spot * (spot[k] * power) / float(cfg.n_mpc))
        Q_spot += l_spot

        # update state variables for prediction
        wall_pred = cfg.wall_mpc(wall, room, tout[k], cfg.thetam)
        room_pred = cfg.room_mpc(wall, room, tout[k], power, cfg.thetam)

        wall = wall_pred
        room = room_pred



    Q_comfort = Function('Q_comfort',
                         [room0, wall0, tset, tdesired, tmin, tout, cfg.thetal, cfg.thetam], [Q_comfort])
    Q_spot = Function('Q_spot',
                         [room0, wall0, tset, tout, spot, cfg.thetal, cfg.thetam], [Q_spot])

    return Q_comfort, Q_spot


def get_categorized_Qbase():
    # measured states
    roommeas = MX.sym('roommeas', cfg.n_rl)
    wallmeas = MX.sym('wallmeas', cfg.n_rl)
    powermeas = MX.sym('powermeas', cfg.n_rl)
    tset = MX.sym('tset', cfg.n_rl)
    dtset = MX.sym('dtset', cfg.n_rl)
    tout = MX.sym('tout', cfg.n_rl)
    tdesired = MX.sym('tdesired', cfg.n_rl)
    tmin = MX.sym('tmin', cfg.n_rl)
    spot = MX.sym('spot', cfg.n_rl)

    Q_comfort_base = 0
    Q_spot_base = 0

    for k in range(cfg.n_rl - 1):
        l_comfort = (cfg.w_tabove * (tdesired[k] - roommeas[k]) ** 2 / float(cfg.n_rl))
        l_comfort += (cfg.w_tbelow * fmax((roommeas[k] - tdesired[k]), 0) ** 2 / float(cfg.n_rl))
        l_comfort += (cfg.w_tmin * fmax((tmin[k] - roommeas[k]), 0) ** 2 / float(cfg.n_rl))

        Q_comfort_base += l_comfort

        l_spot = (cfg.w_spot * (spot[k] * powermeas[k]) / float(cfg.n_mpc))
        Q_spot_base += l_spot

    Q_comfort_base_func = Function('Q_comfort_base_func',
                          (tdesired, roommeas, tmin),
                          [Q_comfort_base])
    Q_spot_base_func = Function('Q_spot_base_func',
                          (powermeas, spot),
                          [Q_spot_base])

    return Q_comfort_base_func, Q_spot_base_func


# get data in order
# baseline1 = './results/Sanity_checks/01_all_noise_basecase.pkl'
# baseline2 = './results/Sanity_checks/01_September_newthetal.pkl'
#

baseline1 = '../results/04_test.pkl'
baseline2 = '../results/04_test_theta.pkl'

print('baseline1', baseline1)
print('baseline2', baseline2)

# baseline1 = './results/02_September_basecase.pkl'
# baseline2 = './results/02_September_newthetal.pkl'

thetal_num0 = np.array([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
thetam_num0 = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0])

# thetal_num = np.array([0.0, 1.0, 0.0, 2.277358890246708, 0.0])
# thetal_num = np.array([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, -0.73651346])

thetal_num = np.array([ 1.76685660e-03, 2.20105263e-07 , 2.14696504e-01,  8.04697005e-01,
  9.94552040e-01, 9.99369196e-01, 6.15629478e-06, -1.51051424e-01
 -6.12682741e-01])

df_history1 = read_results(baseline1)
df_history2 = read_results(baseline2)
t_set_num1, dt_set_num1, t_out_num1, t_desired_num1, t_min_num1, spot_num1, \
room_num1, power_num1, wall_num1, room_noise1, power_noise1 = get_numerical_data(df_history1)

t_set_num2, dt_set_num2, t_out_num2, t_desired_num2, t_min_num2, spot_num2, \
room_num2, power_num2, wall_num2, room_noise2, power_noise2 = get_numerical_data(df_history2)

#build symbolic casadi function to evaluate predicted Q by one (1) MPC iteration
Qmpc = get_Qmpc()
# to test: Qmpc(12, 12, 0, dt_set_num[0:288], t_set_num[0:288], t_desired_num[0:288], t_min_num[0:288], t_out_num[0:288], spot_num[0:288], cfg.thetal_num, cfg.thetam_num)
Qbase = get_Qbase()
# to test: Qbase(t_desired_num[0:cfg.n_rl], room_num[0:cfg.n_rl], t_min_num[0:cfg.n_rl], dt_set_num[0:cfg.n_rl], power_num[0:cfg.n_rl], spot_num[0:cfg.n_rl])
# batches = create_batches(df_history)
batch = list(range(1, len(df_history1) - 1 * max(cfg.n_rl, cfg.n_mpc)))

#difference in cost between unmatched  and matched (theta) MPC and baseline cost in run 1 action sequence
dQ_1 = []
dQ_theta_1 = []

# difference in cost between unmatched and matched (theta) MPC and baseline cost in run 2 action sequence
dQ_2 = []
dQ_theta_2 = []

# comfort cost in untuned mpc
dQ_comfort_2 = []
dQ_comfort_theta_2 = []

# comfort cost in untuned mpc
dQ_comfort_2 = []
dQ_comfort_theta_2 = []

# spot cost in untuned mpc
dQ_spot_2 = []
dQ_spot_theta_2 = []

#### get categorized base cost
Qmpc_comfort, Qmpc_spot = get_catergorized_Qmpc()
Qbase_comfort, Qbase_spot = get_categorized_Qbase()
#
# ### this is the squared residual plots
# for ts in batch:
#     dQ_1.append(  (0.5 * (Qmpc(room_num1[ts], wall_num1[ts],
#                  t_set_num1[ts:ts + cfg.n_mpc], t_desired_num1[ts:ts + cfg.n_mpc],
#                  t_min_num1[ts:ts + cfg.n_mpc],
#                  t_out_num1[ts:ts + cfg.n_mpc], spot_num1[ts:ts + cfg.n_mpc], thetal_num0,  thetam_num0)
#             - Qbase(t_desired_num1[ts:ts + cfg.n_rl], room_num1[ts:ts + cfg.n_rl],
#                     t_min_num1[ts:ts + cfg.n_rl], power_num1[ts+1:ts+1 + cfg.n_rl],
#                     spot_num1[ts:ts + cfg.n_rl])) ** 2 / len(batch)).full().flatten()[0])
#
#     dQ_theta_1.append(( 0.5 * (Qmpc(room_num1[ts], wall_num1[ts],
#                  t_set_num1[ts:ts + cfg.n_mpc], t_desired_num1[ts:ts + cfg.n_mpc],
#                  t_min_num1[ts:ts + cfg.n_mpc],
#                  t_out_num1[ts:ts + cfg.n_mpc], spot_num1[ts:ts + cfg.n_mpc], thetal_num,  thetam_num0)
#             - Qbase(t_desired_num1[ts:ts + cfg.n_rl], room_num1[ts:ts + cfg.n_rl],
#                     t_min_num1[ts:ts + cfg.n_rl], power_num1[ts+1:ts+1 + cfg.n_rl],
#                     spot_num1[ts:ts + cfg.n_rl])) ** 2 / len(batch)).full().flatten()[0])
#
#     dQ_2.append((0.5 * (Qmpc(room_num2[ts], wall_num2[ts],
#                  t_set_num2[ts:ts + cfg.n_mpc], t_desired_num2[ts:ts + cfg.n_mpc],
#                  t_min_num2[ts:ts + cfg.n_mpc],
#                  t_out_num2[ts:ts + cfg.n_mpc], spot_num2[ts:ts + cfg.n_mpc], thetal_num0,  thetam_num0)
#             - Qbase(t_desired_num2[ts:ts + cfg.n_rl], room_num2[ts:ts + cfg.n_rl],
#                     t_min_num2[ts:ts + cfg.n_rl], power_num2[ts+1:ts+1 + cfg.n_rl],
#                     spot_num2[ts:ts + cfg.n_rl])) ** 2 / len(batch)).full().flatten()[0])
# #
#     dQ_theta_2.append((0.5 * (Qmpc(room_num2[ts], wall_num2[ts],
#                         t_set_num2[ts:ts + cfg.n_mpc], t_desired_num2[ts:ts + cfg.n_mpc],
#                         t_min_num2[ts:ts + cfg.n_mpc],
#                         t_out_num2[ts:ts + cfg.n_mpc], spot_num2[ts:ts + cfg.n_mpc], thetal_num, thetam_num0)
#                    - Qbase(t_desired_num2[ts:ts + cfg.n_rl], room_num2[ts:ts + cfg.n_rl],
#                            t_min_num2[ts:ts + cfg.n_rl], power_num2[ts + 1:ts + 1 + cfg.n_rl],
#                            spot_num2[ts:ts + cfg.n_rl])) ** 2 / len(batch)).full().flatten()[0])
#
#     dQ_comfort_2.append((0.5 * (Qmpc_comfort(room_num2[ts], wall_num2[ts],
#                  t_set_num2[ts:ts + cfg.n_mpc], t_desired_num2[ts:ts + cfg.n_mpc],
#                  t_min_num2[ts:ts + cfg.n_mpc],
#                  t_out_num2[ts:ts + cfg.n_mpc], thetal_num0,  thetam_num0)
#             - Qbase_comfort(t_desired_num2[ts:ts + cfg.n_rl], room_num2[ts:ts + cfg.n_rl],
#                     t_min_num2[ts:ts + cfg.n_rl])) ** 2 / len(batch)).full().flatten()[0])
#
#     dQ_comfort_theta_2.append((0.5 * (Qmpc_comfort(room_num2[ts], wall_num2[ts],
#                                              t_set_num2[ts:ts + cfg.n_mpc], t_desired_num2[ts:ts + cfg.n_mpc],
#                                              t_min_num2[ts:ts + cfg.n_mpc],
#                                              t_out_num2[ts:ts + cfg.n_mpc], thetal_num, thetam_num0)
#                                 - Qbase_comfort(t_desired_num2[ts:ts + cfg.n_rl], room_num2[ts:ts + cfg.n_rl],
#                                                 t_min_num2[ts:ts + cfg.n_rl])) ** 2 / len(batch)).full().flatten()[0])
#
#     dQ_spot_2.append((0.5 * (Qmpc_spot(room_num2[ts], wall_num2[ts], t_set_num2[ts:ts + cfg.n_mpc],
#                                        t_out_num2[ts:ts + cfg.n_mpc], spot_num2[ts:ts + cfg.n_mpc],
#                                        thetal_num0, thetam_num0)
#                              - Qbase_spot(power_num2[ts + 1:ts + 1 + cfg.n_rl],
#                            spot_num2[ts:ts + cfg.n_rl])) ** 2 / len(batch)).full().flatten()[0])
#
#     dQ_spot_theta_2.append((0.5 * (Qmpc_spot(room_num2[ts], wall_num2[ts], t_set_num2[ts:ts + cfg.n_mpc],
#                                        t_out_num2[ts:ts + cfg.n_mpc], spot_num2[ts:ts + cfg.n_mpc],
#                                        thetal_num, thetam_num0)
#                              - Qbase_spot(power_num2[ts + 1:ts + 1 + cfg.n_rl],
#                                              spot_num2[ts:ts + cfg.n_rl])) ** 2 / len(batch)).full().flatten()[0])
#

# bins = np.linspace(min(min(dQ_1), min(dQ_theta_1)), max(max(dQ_1), max(dQ_theta_1)), 100)
# plt.hist(dQ_1, bins, alpha=0.5, label='unmatched, training')
# plt.hist(dQ_theta_1, bins, alpha=0.5, label='matched training')
# plt.legend(loc='upper right')
# plt.show()
#
# bins = np.linspace(min(min(dQ_2), min(dQ_theta_2)), max(max(dQ_2), max(dQ_theta_2)), 100)
# plt.hist(dQ_2, bins, alpha=0.5, label='unmatched, test')
# plt.hist(dQ_theta_2, bins, alpha=0.5, label='matched test')
# plt.legend(loc='upper right')
# plt.show()
#
# plt.subplot(4, 1, 1)
# plt.plot(batch, dQ_2, label='dQ untuned MPC')
# plt.plot(batch, dQ_theta_2, label='dQ tuned MPC')
# plt.yscale("log")
# plt.grid("on")
#
# # plot 2:
# plt.subplot(4, 1, 2)
# plt.plot(batch, spot_num2[batch], label='spot', color = 'green')
# plt.legend()
# plt.grid("on")
#
# plt.subplot(4, 1, 3)
# plt.plot(batch, room_noise2[batch], label='room noise', color = 'red')
# plt.legend()
# plt.grid("on")
#
# plt.subplot(4, 1, 4)
# plt.plot(batch, power_noise2[batch], label='power noise')
# plt.legend()
# plt.grid("on")
#
#
# plt.tight_layout()
# plt.show()

# plt.plot(batch, dQ_1, label='dQ untuned MPC')
# plt.plot(batch, dQ_theta_1, label='dQ tuned MPC')
# plt.legend()
# plt.yscale("log")
# plt.grid("on")
# plt.show()
#
# plt.plot(batch, dQ_2, label='dQ untuned MPC')
# plt.plot(batch, dQ_theta_2, label='dQ tuned MPC')
# plt.yscale("log")
# plt.legend()
# plt.grid("on")
# plt.show()

# plt.subplot(3, 1, 1)
# plt.plot(batch, dQ_2, label='dQ untuned MPC')
# plt.plot(batch, dQ_theta_2, label='dQ tuned MPC')
# plt.legend()
# plt.grid("on")
#
# plt.subplot(3, 1, 2)
# plt.plot(batch, dQ_comfort_2, label='comfort untuned')
# plt.plot(batch, dQ_comfort_theta_2, label='comfort tuned')
# plt.legend()
# plt.grid("on")
#
# plt.subplot(3, 1, 3)
# plt.plot(batch, dQ_spot_2, label='spot untuned')
# plt.plot(batch, dQ_spot_theta_2, label='spot tuned')
# plt.legend()
# plt.grid("on")
# plt.tight_layout()
# plt.show()
# #
# # plot trajectories in regions of high cost:
# ts = dQ_theta_2.index(max(dQ_theta_2))
# print(ts, dQ_theta_2[ts])
# ts += 1
# print('Qbase', Qbase(t_desired_num2[ts:ts + cfg.n_rl], room_num2[ts:ts + cfg.n_rl],
#                            t_min_num2[ts:ts + cfg.n_rl], power_num2[ts + 1:ts + 1 + cfg.n_rl],
#                            spot_num2[ts:ts + cfg.n_rl]).full().flatten()[0])
# print('Qmpc', Qmpc(room_num2[ts], wall_num2[ts],
#                         t_set_num2[ts:ts + cfg.n_mpc], t_desired_num2[ts:ts + cfg.n_mpc],
#                         t_min_num2[ts:ts + cfg.n_mpc],
#                         t_out_num2[ts:ts + cfg.n_mpc], spot_num2[ts:ts + cfg.n_mpc],
#                     thetal_num, thetam_num0).full().flatten()[0])
#
# plot_open_loop_at_timestep(ts, df_history2, thetam_num0)
#
# # plot trajectories in regions of low cost:
# ts = dQ_theta_2.index(min(dQ_theta_2))
# ts += 1
# print(ts, dQ_theta_2[ts])
# print('Qbase', Qbase(t_desired_num2[ts:ts + cfg.n_rl], room_num2[ts:ts + cfg.n_rl],
#                      t_min_num2[ts:ts + cfg.n_rl], power_num2[ts + 1:ts + 1 + cfg.n_rl],
#                      spot_num2[ts:ts + cfg.n_rl]).full().flatten()[0])
# print('Qmpc', Qmpc(room_num2[ts], wall_num2[ts],
#                    t_set_num2[ts:ts + cfg.n_mpc], t_desired_num2[ts:ts + cfg.n_mpc],
#                    t_min_num2[ts:ts + cfg.n_mpc],
#                    t_out_num2[ts:ts + cfg.n_mpc], spot_num2[ts:ts + cfg.n_mpc],
#                    thetal_num, thetam_num0).full().flatten()[0])
# plot_open_loop_at_timestep(ts, df_history2, thetam_num0)
#
# print('dQ1', sum(dQ_1))
# print('dQ_matched1', sum(dQ_theta_1))
# print('dQ2', sum(dQ_2))
# print('dQ_matched2', sum(dQ_theta_2))
dQ_1_abs = []
dQ_2_abs = []
dQ_theta_1_abs = []
dQ_theta_2_abs = []
## residuals
for ts in batch:
    dQ_1_abs.append( (Qmpc(room_num1[ts], wall_num1[ts],
                 t_set_num1[ts:ts + cfg.n_mpc], t_desired_num1[ts:ts + cfg.n_mpc],
                 t_min_num1[ts:ts + cfg.n_mpc],
                 t_out_num1[ts:ts + cfg.n_mpc], spot_num1[ts:ts + cfg.n_mpc], thetal_num0,  thetam_num0)
            - Qbase(t_desired_num1[ts:ts + cfg.n_rl], room_num1[ts:ts + cfg.n_rl],
                    t_min_num1[ts:ts + cfg.n_rl], power_num1[ts+1:ts+1 + cfg.n_rl],
                    spot_num1[ts:ts + cfg.n_rl])).full().flatten()[0] / len(batch))

    dQ_theta_1_abs.append((Qmpc(room_num1[ts], wall_num1[ts],
                 t_set_num1[ts:ts + cfg.n_mpc], t_desired_num1[ts:ts + cfg.n_mpc],
                 t_min_num1[ts:ts + cfg.n_mpc],
                 t_out_num1[ts:ts + cfg.n_mpc], spot_num1[ts:ts + cfg.n_mpc], thetal_num,  thetam_num0)
            - Qbase(t_desired_num1[ts:ts + cfg.n_rl], room_num1[ts:ts + cfg.n_rl],
                    t_min_num1[ts:ts + cfg.n_rl], power_num1[ts+1:ts+1 + cfg.n_rl],
                    spot_num1[ts:ts + cfg.n_rl]) ).full().flatten()[0] / len(batch))

    dQ_2_abs.append((Qmpc(room_num2[ts], wall_num2[ts],
                 t_set_num2[ts:ts + cfg.n_mpc], t_desired_num2[ts:ts + cfg.n_mpc],
                 t_min_num2[ts:ts + cfg.n_mpc],
                 t_out_num2[ts:ts + cfg.n_mpc], spot_num2[ts:ts + cfg.n_mpc],
                          thetal_num0,  thetam_num0)
            - Qbase(t_desired_num2[ts:ts + cfg.n_rl], room_num2[ts:ts + cfg.n_rl],
                    t_min_num2[ts:ts + cfg.n_rl], power_num2[ts+1:ts+1 + cfg.n_rl],
                    spot_num2[ts:ts + cfg.n_rl])).full().flatten()[0] / len(batch))
#
    dQ_theta_2_abs.append((Qmpc(room_num2[ts], wall_num2[ts],
                        t_set_num2[ts:ts + cfg.n_mpc], t_desired_num2[ts:ts + cfg.n_mpc],
                        t_min_num2[ts:ts + cfg.n_mpc],
                        t_out_num2[ts:ts + cfg.n_mpc], spot_num2[ts:ts + cfg.n_mpc], thetal_num, thetam_num0)
                   - Qbase(t_desired_num2[ts:ts + cfg.n_rl], room_num2[ts:ts + cfg.n_rl],
                           t_min_num2[ts:ts + cfg.n_rl], power_num2[ts + 1:ts + 1 + cfg.n_rl],
                           spot_num2[ts:ts + cfg.n_rl])).full().flatten()[0] / len(batch))

plt.plot(batch, dQ_1_abs, label='dQ untuned MPC')
plt.plot(batch, dQ_theta_1_abs, label='dQ tuned MPC')
plt.legend()
plt.grid("on")
plt.show()

plt.plot(batch, dQ_2_abs, label='dQ untuned MPC')
plt.plot(batch, dQ_theta_2_abs, label='dQ tuned MPC')
plt.legend()
plt.grid("on")
plt.show()


bins = np.linspace(min(min(dQ_1_abs),min(dQ_theta_1_abs)), max(max(dQ_1_abs),max(dQ_theta_1_abs)), 100)
plt.hist(dQ_1_abs, bins, alpha=0.5, label='unmatched, training')
plt.hist(dQ_theta_1_abs, bins, alpha=0.5, label='matched training')
plt.legend(loc='upper right')
plt.show()

bins = np.linspace(min(min(dQ_2_abs), min(dQ_theta_2_abs)), max(max(dQ_2_abs), max(dQ_theta_2_abs)), 100)
plt.hist(dQ_2_abs, bins, alpha=0.5, label='unmatched, test')
plt.hist(dQ_theta_2_abs, bins, alpha=0.5, label='matched test')
plt.legend(loc='upper right')
plt.show()

print('Residuals untuned MPC, training action sequence: ', sum([np.abs(i) for i in dQ_1_abs]) )
print('Residuals tuned MPC, training action sequence: ', sum([np.abs(i) for i in dQ_theta_1_abs]))
print('Residuals untuned MPC, test action sequence: ', sum([np.abs(i) for i in dQ_2_abs]) )
print('Residuals tuned MPC, test action sequence: ', sum([np.abs(i) for i in dQ_theta_2_abs]))

