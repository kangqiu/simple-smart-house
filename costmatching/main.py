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

def get_mpc_prediction():
    pass
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
        room_pred = cfg.room_mpc(wall, room, tout[k], power, cfg.thetam)

        wall = wall_pred
        room = room_pred

    t_mpc = cfg.tmpc_func(room[-1], tdesired[-1], tmin[-1], cfg.thetat)
    Qmpc += t_mpc
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

def check_gradient(dQ, Q, room0, wall0, power0, dt_set, t_set, t_desired, t_min, t_out, spot, gradient_type):

    k = 0
    Q_num = float(Q(room0, wall0, power0, dt_set, t_set,t_desired, t_min, t_out, spot, cfg.thetal_num,
                       cfg.thetam_num, cfg.thetat_num))

    dQ_num = dQ(room0, wall0, power0, dt_set, t_set,t_desired, t_min, t_out, spot, cfg.thetal_num,
                       cfg.thetam_num, cfg.thetat_num).full().flatten()
    delta_Q = []
    error = {}

    n = 50
    if gradient_type == 'l':
        theta = cfg.thetal_num
    elif gradient_type == 'm':
        theta = cfg.thetam_num

    for ind in range(len(theta)):
        error[ind] = []

    for i in np.linspace(0, 10, n):
        perturbation = np.full(np.shape(theta), 1) * i * 0.01
        theta_perturbed = theta + perturbation

        if gradient_type == 'l':
            q_per = float(Q(room0, wall0, power0, dt_set, t_set,t_desired, t_min, t_out, spot, theta_perturbed,
                           cfg.thetam_num, cfg.thetat_num).full().flatten())

        elif gradient_type == 'm':
            q_per = float(Q(room0, wall0, power0, dt_set, t_set, t_desired, t_min, t_out, spot, cfg.thetal_num,
                            theta_perturbed, cfg.thetat_num).full().flatten())
        dq = dQ_num * (theta_perturbed - theta)

        delta_Q.append(q_per - Q_num)
        for ind in range(len(theta)):
            error[ind].append(dq[ind] - (q_per - Q_num))

    for ind in range(len(theta)):
        plt.plot(np.linspace(0, 10, n), error[ind])
        plt.grid('on')
        plt.title( str(ind))
        plt.show()

    return

# def constrained_w_update(dQ, theta, alpha):
#     w = theta
#     param = MX.sym('param', 2, len(dQ))
#     param_num = np.stack((dQ, theta))
#     J = 0.5 * dot(w, w) / alpha['w'] + dot(param[0, :].T, w)
#
#     g = sum1(param[1, :].T + w)
#     lbg = 1
#     ubg = 1
#
#     LS = {'f': J, 'x': w, 'g': g, 'p': param}
#     options = {'print_time': 0}
#     options['ipopt'] = {'linear_solver': 'ma27',
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
    # batches = create_batches(df_history)
    batch = list(range(len(df_history) - 2 * max(cfg.n_rl, cfg.n_mpc)))

    if cfg.use_ipopt == True:
        w = MX.sym('w', int(len(cfg.thetal_num) + len(cfg.thetat_num) + len(cfg.thetam_num)))

        J = 0
        for ts in batch:
            J += 0.5 * (Qmpc(room_num[ts], wall_num[ts], power_num[ts], dt_set_num[ts:ts + cfg.n_mpc],
                             t_set_num[ts:ts + cfg.n_mpc], t_desired_num[ts:ts + cfg.n_mpc],
                             t_min_num[ts:ts + cfg.n_mpc],
                             t_out_num[ts:ts + cfg.n_mpc], spot_num[ts:ts + cfg.n_mpc], w[0:6], w[9:19], w[6:9])
                        - Qbase(t_desired_num[ts:ts + cfg.n_rl], room_num[ts:ts + cfg.n_rl],
                                t_min_num[ts:ts + cfg.n_rl],
                                dt_set_num[ts:ts + cfg.n_rl], power_num[ts:ts + cfg.n_rl],
                                spot_num[ts:ts + cfg.n_rl])) ** 2 / len(batch)
        #
        ubw = [+inf] * int(len(cfg.thetal_num) + len(cfg.thetat_num) + len(cfg.thetam_num))
        lbw = [0] * int(len(cfg.thetal_num) + len(cfg.thetat_num) + len(cfg.thetam_num))

        g = []
        lbg = []
        ubg = []
        #
        LS = {'f': J, 'x': w, 'g': g, 'p': []}
        options = {'print_time': 0}
        options['ipopt'] = {'linear_solver': 'ma57',
                            'max_iter': 200}
        solverLS = nlpsol('solver', 'ipopt', LS, options)
        w0 = d=np.concatenate((cfg.thetal_num, cfg.thetat_num, cfg.thetam_num), axis=None)

        sol = solverLS(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=[])
        w_opt = sol['x'].full().flatten()

        print(w_opt)


    if cfg.use_ipopt == False:
        # nest into episodes
        loss_over_episodes = []

        ## check gradient
        # ts = 0
        # check_gradient(dQmpc_m, Qmpc, room_num[ts], wall_num[ts], power_num[ts], dt_set_num[ts:ts+cfg.n_mpc],
        #                 t_set_num[ts:ts+cfg.n_mpc],t_desired_num[ts:ts+cfg.n_mpc], t_min_num[ts:ts+cfg.n_mpc],
        #                 t_out_num[ts:ts+cfg.n_mpc], spot_num[ts:ts+cfg.n_mpc], 'm')

        for e, index in enumerate(tqdm(range(cfg.episodes))):
            #iterate through batches
            loss_per_batch = 0
            dthetal_batch = 0
            dthetam_batch = 0
            # dthetat_batch = 0
            for ts in batch:
                #collect Qmpc per timestep in batch
                qmpc = Qmpc(room_num[ts], wall_num[ts], power_num[ts], dt_set_num[ts:ts+cfg.n_mpc],
                            t_set_num[ts:ts+cfg.n_mpc],t_desired_num[ts:ts+cfg.n_mpc], t_min_num[ts:ts+cfg.n_mpc],
                            t_out_num[ts:ts+cfg.n_mpc], spot_num[ts:ts+cfg.n_mpc], thetal_num, thetam_num,thetat_num )
                qbase = Qbase(t_desired_num[ts:ts+cfg.n_rl], room_num[ts:ts+cfg.n_rl], t_min_num[ts:ts+cfg.n_rl],
                                     dt_set_num[ts:ts+cfg.n_rl], power_num[ts:ts+cfg.n_rl], spot_num[ts:ts+cfg.n_rl])
                loss = (0.5*(qmpc-qbase)**2).full().flatten()[0]/len(batch)

                dQl = dQmpc_l(room_num[ts], wall_num[ts], power_num[ts], dt_set_num[ts:ts + cfg.n_mpc],
                              t_set_num[ts:ts + cfg.n_mpc], t_desired_num[ts:ts + cfg.n_mpc], t_min_num[ts:ts + cfg.n_mpc],
                              t_out_num[ts:ts + cfg.n_mpc], spot_num[ts:ts + cfg.n_mpc], thetal_num, thetam_num, thetat_num).full().flatten()
                # dQm = dQmpc_m(room_num[ts], wall_num[ts], power_num[ts], dt_set_num[ts:ts + cfg.n_mpc],
                #               t_set_num[ts:ts + cfg.n_mpc], t_desired_num[ts:ts + cfg.n_mpc],
                #               t_min_num[ts:ts + cfg.n_mpc],
                #               t_out_num[ts:ts + cfg.n_mpc], spot_num[ts:ts + cfg.n_mpc], thetal_num, thetam_num,
                #               thetat_num).full().flatten()
                # dQt = dQmpc_t(room_num[ts], wall_num[ts], power_num[ts], dt_set_num[ts:ts + cfg.n_mpc],
                #               t_set_num[ts:ts + cfg.n_mpc], t_desired_num[ts:ts + cfg.n_mpc],
                #               t_min_num[ts:ts + cfg.n_mpc],
                #               t_out_num[ts:ts + cfg.n_mpc], spot_num[ts:ts + cfg.n_mpc], thetal_num, thetam_num,
                #               thetat_num).full().flatten()

                dthetal_batch += loss * dQl
                # dthetam_batch += loss * dQm
                # dthetat_batch -= loss * dQt
                loss_per_batch += loss

            # TODO: why the fuck does it converge in the positive?
            thetal_num -= cfg.alphal * dthetal_batch
            # thetam_num += cfg.alpham * dthetam_batch
            # thetat_num += cfg.alphat * dthetat_batch

            loss_over_episodes.append(loss_per_batch)


        print(thetal_num)
        print(thetam_num)
        print(thetat_num)
        plt.plot(range(cfg.episodes), loss_over_episodes)
        plt.grid('on')
        plt.show()




if __name__ == '__main__':
    main()
