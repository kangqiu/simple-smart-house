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
    # external params
    t_out_num = df_history['t_out'].values
    t_desired_num = df_history['t_desired'].values
    t_min_num = df_history['t_min'].values
    spot_num = df_history['spot_price'].values

    return t_set_num, t_out_num, t_desired_num, t_min_num, spot_num, room_num, power_num, wall_num

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
                         [room0, wall0, tset, tdesired, tmin, tout, spot, cfg.thetal, cfg.thetam],
                         [Qmpc])
    return Qmpc_func

def get_gradients(Qmpc):
    dQ_l = Qmpc.factory('dQmpc_func', ['i0', 'i1', 'i2', 'i3', 'i4', 'i5', 'i6', 'i7', 'i8'], ['jac:o0:i7'])
    dQ_m = Qmpc.factory('dQmpc_func', ['i0', 'i1', 'i2', 'i3', 'i4', 'i5', 'i6', 'i7', 'i8'], ['jac:o0:i8'])

    return dQ_l, dQ_m

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
    for k in range(cfg.n_rl-1):
        l_meas = cfg.lmeas_func(tdesired[k], roommeas[k], tmin[k], powermeas[k], spot[k])
        Qbase += l_meas

    Qbase += (cfg.w_tbelow * fmax((roommeas[cfg.n_rl-1] - tdesired[cfg.n_rl-1]), 0) ** 2 / float(cfg.n_rl))
    Qbase += (cfg.w_tmin * fmax((tmin[cfg.n_rl-1] - roommeas[cfg.n_rl-1 ]),0) ** 2 / float(cfg.n_rl))

    Qbase_func = Function('Qbase_func',
                         (tdesired, roommeas, tmin, powermeas, spot),
                         [Qbase])

    return Qbase_func

def check_gradient(dQl, dQm, Q, room0, wall0, t_set, t_desired, t_min, t_out, spot, gradient_type):

    k = 0
    Q_num = float(Q(room0, wall0, t_set, t_desired, t_min, t_out, spot, cfg.thetal_num,
                       cfg.thetam_num))
    delta_Q = []
    error = {}

    n = 50
    if gradient_type == 'l':
        theta = cfg.thetal_num
        dQ_num = dQl(room0, wall0, t_set, t_desired, t_min, t_out, spot, cfg.thetal_num,
                    cfg.thetam_num).full().flatten()
    elif gradient_type == 'm':
        theta = cfg.thetam_num
        dQ_num = dQm(room0, wall0, t_set, t_desired, t_min, t_out, spot, cfg.thetal_num,
                    cfg.thetam_num).full().flatten()

    for ind in range(len(theta)):
        error[ind] = []

    for i in np.linspace(0, 10, n):
        perturbation = np.full(np.shape(theta), 1) * i * 0.01
        theta_perturbed = theta + perturbation

        if gradient_type == 'l':
            q_per = float(Q(room0, wall0, t_set,t_desired, t_min, t_out, spot, theta_perturbed,
                           cfg.thetam_num).full().flatten())

        elif gradient_type == 'm':
            q_per = float(Q(room0, wall0, power0, t_set, t_desired, t_min, t_out, spot, cfg.thetal_num,
                            theta_perturbed).full().flatten())
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


def main():
    print(cfg.results_file)
    print('tuning of slackmin location, regularized at 1e-2')
    # get data in order
    thetal_num = cfg.thetal_num
    thetam_num = cfg.thetam_num

    df_history = datahandling.read_results()
    t_set_num, t_out_num, t_desired_num, t_min_num, spot_num, room_num, power_num, wall_num= get_numerical_data(df_history)

    #build symbolic casadi function to evaluate predicted Q by one (1) MPC iteration
    Qmpc = get_Qmpc()
    dQl, dQm = get_gradients(Qmpc)

    # ts = 5
    # check_gradient(dQl, dQm, Qmpc, room_num[ts], wall_num[ts],
    #                t_set_num[ts:ts+cfg.n_mpc],t_desired_num[ts:ts+cfg.n_mpc], t_min_num[ts:ts+cfg.n_mpc],
    #                t_out_num[ts:ts+cfg.n_mpc], spot_num[ts:ts+cfg.n_mpc], 'l')

    Qbase = get_Qbase()
    # to test: Qbase(t_desired_num[0:cfg.n_rl], room_num[0:cfg.n_rl], t_min_num[0:cfg.n_rl], dt_set_num[0:cfg.n_rl], power_num[0:cfg.n_rl], spot_num[0:cfg.n_rl])
    # batches = create_batches(df_history)
    batch = list(range(1, len(df_history) - 1 * max(cfg.n_rl, cfg.n_mpc)))

    if cfg.use_ipopt:
        # w = MX.sym('w', int(len(cfg.thetal_num)))
        w = MX.sym('w', 9)
        thetal = cfg.thetal_num
        J = 0
        for ts in batch:

            # # check if stage cost functions are correct -> they are
            # print(cfg.lmpc_func(t_desired_num[ts], t_min_num[ts], room_num[ts], power_num[ts], spot_num[ts], cfg.thetal_num))
            # print(cfg.lmeas_func(t_desired_num[ts], room_num[ts], t_min_num[ts], power_num[ts], spot_num[ts]))
            #
            # # check if state predictions are correct
            # power = cfg.power_mpc(room_num[ts], t_set_num[ts+1], cfg.thetam_num)
            # print(power, power_num[ts+1])
            #
            # wall_pred = cfg.wall_mpc(wall_num[ts], room_num[ts], t_out_num[ts], cfg.thetam_num)
            # print(wall_pred, wall_num[ts+1])
            # room_pred = cfg.room_mpc(wall_num[ts], room_num[ts], t_out_num[ts], power, cfg.thetam_num)
            #
            # print(room_pred, room_num[ts+1])
            #
            #
            # print(Qmpc(room_num[ts], wall_num[ts],
            #      t_set_num[ts:ts + cfg.n_mpc], t_desired_num[ts:ts + cfg.n_mpc],
            #      t_min_num[ts:ts + cfg.n_mpc],
            #      t_out_num[ts:ts + cfg.n_mpc], spot_num[ts:ts + cfg.n_mpc], cfg.thetal_num, cfg.thetam_num),
            # Qbase(t_desired_num[ts:ts + cfg.n_rl], room_num[ts:ts + cfg.n_rl],
            #       t_min_num[ts:ts + cfg.n_rl],
            #       power_num[ts+1:ts+1 + cfg.n_rl],
            #       spot_num[ts:ts + cfg.n_rl]))

            res = (Qmpc(room_num[ts], wall_num[ts],
                             t_set_num[ts:ts + cfg.n_mpc], t_desired_num[ts:ts + cfg.n_mpc],
                             t_min_num[ts:ts + cfg.n_mpc],
                             t_out_num[ts:ts + cfg.n_mpc], spot_num[ts:ts + cfg.n_mpc], w,  cfg.thetam_num)
                  - Qbase(t_desired_num[ts:ts + cfg.n_rl], room_num[ts:ts + cfg.n_rl],
                                t_min_num[ts:ts + cfg.n_rl],
                                power_num[ts+1:ts+1 + cfg.n_rl],
                                spot_num[ts:ts + cfg.n_rl]))

            J += 0.5 * (res) ** 2 / len(batch)
            # J += (0.5 ** 2) * (sqrt(1 + (res/0.5) ** 2) - 1) / len(batch)

        # #add regularization to slack variable tuning
        for t in range(len(cfg.thetal_num)):
            J += 1e-2 * (w[t] - cfg.thetal_num[t])**2
        #
        ubw = [+inf] * 9
        lbw = [-inf] + [0] * 5 + [-inf]*3
        g = []
        lbg = []
        ubg = []
        #
        LS = {'f': J, 'x': w, 'g': g, 'p': []}
        options = {'print_time': 0}
        options['ipopt'] = {'linear_solver': 'ma57',
                            'max_iter': 50
                            }
        solverLS = nlpsol('solver', 'ipopt', LS, options)
        # w0 = np.concatenate((cfg.thetal_num, cfg.thetam_num), axis=None)
        w0 = cfg.thetal_num

        sol = solverLS(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=[])
        w_opt = sol['x'].full().flatten()

        print(w_opt)


    if cfg.use_ipopt == False:
        # nest into episodes
        loss_over_episodes = []
        theta0 = 0
        thetal = cfg.thetal_num
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
                room = room_num[ts]
                wall = wall_num[ts]
                lmpc = []
                lbase = []

                # for k in range(cfg.n_mpc):
                #     power = cfg.power_mpc(room, t_set_num[ts+k], cfg.thetam_num).full().flatten()
                #
                #     l_mpc = cfg.lmpc_func(t_desired_num[ts+k], t_min_num[ts + k], room, power, spot_num[ts + k], thetal)
                #
                #     # update state variables for prediction
                #     wall_pred = cfg.wall_mpc(wall, room, t_out_num[ts+k], cfg.thetam_num).full().flatten()
                #     room_pred = cfg.room_mpc(wall, room, t_out_num[ts+k], power, cfg.thetam_num).full().flatten()
                #
                #     wall = wall_pred
                #     room = room_pred
                #     lmpc.append(l_mpc.full().flatten()[0])
                # for k in range(cfg.n_mpc):
                #     l_meas = cfg.lmeas_func(t_desired_num[ts+k], room_num[ts+k], t_min_num[ts+k], power_num[ts+k+1], spot_num[ts+k])
                #     lbase.append(l_meas.full().flatten()[0])
                #
                # plt.plot(range(cfg.n_mpc), lmpc, label = 'mpc')
                # plt.plot(range(cfg.n_mpc), lbase, label = 'base')
                # plt.legend()
                # plt.show()
                qmpc = Qmpc(room_num[ts], wall_num[ts],
                         t_set_num[ts:ts + cfg.n_mpc], t_desired_num[ts:ts + cfg.n_mpc],
                         t_min_num[ts:ts + cfg.n_mpc],
                         t_out_num[ts:ts + cfg.n_mpc], spot_num[ts:ts + cfg.n_mpc], thetal,  cfg.thetam_num)
                qbase = Qbase(t_desired_num[ts:ts + cfg.n_rl], room_num[ts:ts + cfg.n_rl],
                            t_min_num[ts:ts + cfg.n_rl],
                            power_num[ts+1:ts+1 + cfg.n_rl],
                            spot_num[ts:ts + cfg.n_rl])

                residual = qmpc-qbase

                if residual < 0:
                    loss = (0.5*(qmpc-qbase)**2).full().flatten()[0]/len(batch)

                    dQ_l = dQl(room_num[ts], wall_num[ts],
                             t_set_num[ts:ts + cfg.n_mpc], t_desired_num[ts:ts + cfg.n_mpc],
                             t_min_num[ts:ts + cfg.n_mpc],
                             t_out_num[ts:ts + cfg.n_mpc], spot_num[ts:ts + cfg.n_mpc], thetal,  cfg.thetam_num).full().flatten()[3]
                    dthetal_batch += loss * dQ_l
                    # dthetam_batch += loss * dQm
                    # dthetat_batch -= loss * dQt
                    loss_per_batch += loss
                else:
                    pass
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

            theta0 += cfg.alphal * dthetal_batch
            thetal[3] = theta0
            # thetam_num += cfg.alpham * dthetam_batch
            # thetat_num += cfg.alphat * dthetat_batch

            loss_over_episodes.append(loss_per_batch)


        print(theta0)
        # print(thetam_num)
        # print(thetat_num)
        plt.plot(range(cfg.episodes), loss_over_episodes)
        plt.grid('on')
        plt.show()



if __name__ == '__main__':
    main()
