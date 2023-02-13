from datetime import date, datetime, timedelta
import pytz
import matplotlib.pyplot as plt
import pandas as pd
from casadi.tools import *
from tqdm import tqdm
from time import sleep
import datahandling
import config_cost as cfg

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

    return t_set_num, dt_set_num, t_out_num, t_desired_num, t_min_num, spot_num, room_num, power_num, wall_num
def get_Qmpc():
    # measured states
    room0 = MX.sym('room0')
    wall0 = MX.sym('wall0')
    power0 = MX.sym('power0')
    # initialize state variables
    room = room0
    wall = wall0
    power = power0

    tset = MX.sym('tset', cfg.n_mpc)
    dtset = MX.sym('dtset', cfg.n_mpc)
    tout = MX.sym('tout', cfg.n_mpc)
    tdesired = MX.sym('tdesired', cfg.n_mpc)
    tmin = MX.sym('tmin', cfg.n_mpc)
    spot = MX.sym('spot', cfg.n_mpc)

    Qmpc = 0
    for k in range(cfg.n_mpc-1):
        l_mpc = cfg.lmpc_func(tdesired[k], tmin[k], room, dtset[k], power, spot[k], cfg.thetal)
        Qmpc += l_mpc

        # update state variables for prediction
        power_unsat = cfg.power_mpc(room, tset[k], cfg.thetam)
        power = cfg.satpower_mpc(power_unsat)
        wall_pred = cfg.wall_mpc(wall, room, tout[k], cfg.thetam)
        room_pred = cfg.room_mpc(wall_pred, room, tout[k], power, cfg.thetam)

        wall = wall_pred
        room = room_pred

    t_mpc = cfg.tmpc_func(room[-1], tdesired[-1], tmin[-1], cfg.thetat)
    Qmpc_func = Function('Qmpc_func',
                         [room0, wall0, power0, dtset, tset, tdesired, tmin, tout, spot, cfg.thetal, cfg.thetam, cfg.thetat],
                         [Qmpc])
    dQmpc_l = Qmpc_func.factory('dQmpc_func', ['i0', 'i1', 'i2', 'i3', 'i4', 'i5', 'i6', 'i7', 'i8',
                                               'i9', 'i10', 'i11'], ['jac:o0:i9'])
    dQmpc_m = Qmpc_func.factory('dQmpc_func', ['i0', 'i1', 'i2', 'i3', 'i4', 'i5', 'i6', 'i7', 'i8',
                                               'i9', 'i10', 'i11'], ['jac:o0:i10'])
    dQmpc_t = Qmpc_func.factory('dQmpc_func', ['i0', 'i1', 'i2', 'i3', 'i4', 'i5', 'i6', 'i7', 'i8',
                                               'i9', 'i10', 'i11'], ['jac:o0:i11'])
    return Qmpc_func, dQmpc_l, dQmpc_m, dQmpc_t

def get_Qbase():
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

    Qbase = 0
    for k in range(cfg.n_rl):
        l_meas = cfg.lmeas_func(tdesired[k], roommeas[k], tmin[k], dtset[k], powermeas[k], spot[k])
        Qbase += cfg.gamma**k * l_meas

    Qbase_func = Function('Qbase_func',
                         (tdesired, roommeas, tmin, dtset, powermeas, spot),
                         [Qbase])

    return Qbase_func

def create_batches(df_history):
    timesteps = list(range(len(df_history) - max(cfg.n_rl, cfg.n_mpc)))
    b_split = list(datahandling.split(timesteps, cfg.n_batches))
    batches = []
    for b in b_split:
        end = b[0] + cfg.len_batches
        if end > timesteps[-1]:
            end = timesteps[-1]
        batch = timesteps[b[0]:end]
        batches.append(batch)

    return batches

# def constrained_w_update(dQ, theta):
#     w = MX.sym('w', 3)
#     param = MX.sym('param', 2, 3)
#     param_num = np.stack((dQ, theta))
#     J = 0.5 * dot(w, w) / self.alpha['w'] + dot(param[0, :].T, w)
#
#     g = sum1(param[1, :].T + w)
#     lbg = 1
#     ubg = 1
#
#     LS = {'f': J, 'x': w, 'g': g, 'p': param}
#     options = {'print_time': 0}
#     options['ipopt'] = {'linear_solver': 'ma27',
#                         'hsllib': '/Users/kqiu/ThirdParty-HSL/.libs/libcoinhsl.dylib',
#                         'print_level': 0}
#     solverLS = nlpsol('solver', 'ipopt', LS, options)
#
#     sol = solverLS(x0=[0, 0, 0], lbg=lbg, ubg=ubg, p=param_num)
#     w_opt = sol['x'].full().flatten()
#
#     return w_opt

def main():
    # get data in order
    thetal_num = cfg.thetal_num
    thetat_num = cfg.thetat_num
    thetam_num = cfg.thetam_num

    df_history = datahandling.read_results()
    t_set_num, dt_set_num, t_out_num, t_desired_num, t_min_num, spot_num, room_num, power_num, wall_num= get_numerical_data(df_history)

    #build symbolic casadi function to evaluate predicted Q by one (1) MPC iteration
    Qmpc, dQmpc_l, dQmpc_m, dQmpc_t = get_Qmpc()
    # to test: Qmpc(12, 12, 0, dt_set_num[0:288], t_set_num[0:288], t_desired_num[0:288], t_min_num[0:288], t_out_num[0:288], spot_num[0:288], cfg.thetal_num, cfg.thetam_num)
    Qbase = get_Qbase()
    # to test: Qbase(t_desired_num[0:cfg.n_rl], room_num[0:cfg.n_rl], t_min_num[0:cfg.n_rl], dt_set_num[0:cfg.n_rl], power_num[0:cfg.n_rl], spot_num[0:cfg.n_rl])

    batches = create_batches(df_history)
    # nest into episodes
    loss_over_episodes = []
    for e, index in enumerate(tqdm(range(cfg.episodes))):
        #iterate through batches
        loss_per_episode = 0
        for b in range(len(batches)):
            batch = batches[b]
            dthetal_batch = 0
            dthetam_batch = 0
            dthetat_batch = 0
            loss_per_batch = 0
            for ts in batch:
                #collect Qmpc per timestep in batch
                qmpc = Qmpc(room_num[ts], wall_num[ts], power_num[ts], dt_set_num[ts:ts+cfg.n_mpc],
                                   t_set_num[ts:ts+cfg.n_mpc],t_desired_num[ts:ts+cfg.n_mpc], t_min_num[ts:ts+cfg.n_mpc],
                                   t_out_num[ts:ts+cfg.n_mpc], spot_num[ts:ts+cfg.n_mpc], thetal_num, thetam_num,thetat_num )
                qbase = Qbase(t_desired_num[ts:ts+cfg.n_rl], room_num[ts:ts+cfg.n_rl], t_min_num[ts:ts+cfg.n_rl],
                                     dt_set_num[ts:ts+cfg.n_rl], power_num[ts:ts+cfg.n_rl], spot_num[ts:ts+cfg.n_rl])
                loss = (0.5*(qmpc-qbase)**2).full().flatten()[0]/ len(batch)

                dQl = dQmpc_l(room_num[ts], wall_num[ts], power_num[ts], dt_set_num[ts:ts + cfg.n_mpc],
                              t_set_num[ts:ts + cfg.n_mpc], t_desired_num[ts:ts + cfg.n_mpc], t_min_num[ts:ts + cfg.n_mpc],
                              t_out_num[ts:ts + cfg.n_mpc], spot_num[ts:ts + cfg.n_mpc], thetal_num, thetam_num, thetat_num).full().flatten()
                dQm = dQmpc_m(room_num[ts], wall_num[ts], power_num[ts], dt_set_num[ts:ts + cfg.n_mpc],
                              t_set_num[ts:ts + cfg.n_mpc], t_desired_num[ts:ts + cfg.n_mpc],
                              t_min_num[ts:ts + cfg.n_mpc],
                              t_out_num[ts:ts + cfg.n_mpc], spot_num[ts:ts + cfg.n_mpc], thetal_num, thetam_num,
                              thetat_num).full().flatten()
                dQt = dQmpc_t(room_num[ts], wall_num[ts], power_num[ts], dt_set_num[ts:ts + cfg.n_mpc],
                              t_set_num[ts:ts + cfg.n_mpc], t_desired_num[ts:ts + cfg.n_mpc],
                              t_min_num[ts:ts + cfg.n_mpc],
                              t_out_num[ts:ts + cfg.n_mpc], spot_num[ts:ts + cfg.n_mpc], thetal_num, thetam_num,
                              thetat_num).full().flatten()
                dthetal_batch -= loss * dQl
                dthetam_batch -= loss * dQm
                dthetat_batch -= loss * dQt
                loss_per_batch += loss
            # TODO: constrained update
            if cfg.constrained_update:
                pass
            else:
                thetal_num += cfg.alphal * dthetal_batch / len(batches)
                thetam_num += cfg.alpham * dthetam_batch / len(batches)
                thetat_num += cfg.alphat * dthetat_batch / len(batches)
            loss_per_episode += loss_per_batch
        loss_over_episodes.append(loss_per_episode)
    print(thetal_num)
    print(thetam_num)
    print(thetat_num)
    plt.plot(range(cfg.episodes), loss_over_episodes)
    plt.grid('on')
    plt.show()







if __name__ == '__main__':
    main()
