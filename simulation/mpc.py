################################################################################################
# package imports
from casadi.tools import *
from matplotlib import pyplot as plt
################################################################################################
# file imports
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)

import datahandling
import formulation as form



def get_model(w, data, thetal):
    g = []
    lbg = []
    ubg = []
    for k in range(form.n_mpc - 1):
        # power prediction
        power = form.power_func(w['state', k, 'room'], w['state', k+1, 't_target'])
        power = form.satpower_func(power)
        # Power Temperature Dynamics
        g.append(power - w['state', k, 'power'])
        lbg.append(0)
        ubg.append(0)

        t_wall_plus = form.wall_func(w['state', k, 'wall'], w['state', k, 'room'], data['t_out', k])
        t_room_plus = form.room_func(w['state', k, 'wall'], w['state', k, 'room'], data['t_out', k], power)

        # Room Temperature Dynamics
        g.append(t_room_plus - w['state', k + 1, 'room'])
        lbg.append(0)
        ubg.append(0)

        # Wall Temperature Dynamics
        g.append(t_wall_plus - w['state', k + 1, 'wall']
                 )
        lbg.append(0)
        ubg.append(0)

        ### Control dynamics ###
        g.append(w['state', k, 't_target'] +
                 w['input', k, 'dt_target']
                 - w['state', k + 1, 't_target'])
        lbg.append(0)
        ubg.append(0)

        ### Slack stuff ###
        g.append(form.slack_mpc(w['state', k, 'room'], data['t_max', k], thetal[7])
                 - w['state', k, 'slack'])
        lbg.append(-inf)
        ubg.append(0)

        g.append(form.slack_mpc(w['state', k, 'room'], data['t_min', k],
                                thetal[8])
                 - w['state', k, 'slackmin'])
        lbg.append(-inf)
        ubg.append(0)



    return g, lbg, ubg

def get_objective(w, data, thetal):
    ### Cost ###
    J = 0

    for k in range(form.n_mpc - 1):
        J += form.lmpc_func(data['t_mid', k] , w['state', k, 'room'], w['state', k, 'slack'],
                           w['state', k, 'slackmin'] , w['input', k, 'dt_target'], w['state', k, 'power'],
                           data['spot', k], thetal)
    # terminal cost
    J += form.tmpc_func(w['state', -1, 'slack'], w['state', -1, 'slackmin'], thetal)
    return J

def instantiate(thetal):
    # get variable MPC Data structure
    data = [entry('t_out', repeat=form.n_mpc),
            entry('spot',  repeat=form.n_mpc),
            entry('t_min', repeat=form.n_mpc),
            entry('t_max', repeat=form.n_mpc),
            entry('t_mid', repeat=form.n_mpc)]
    data = struct_symMX(data)

    #get w
    states = struct_symMX([entry('room'),
                           entry('wall'),
                           entry('slackmin'),
                           entry('slack'),
                           entry('power'),
                           entry('t_target')
                           ])
    # MPC inputs
    inputs = struct_symMX([entry('dt_target')])
    states = struct_symMX(states)
    inputs = struct_symMX(inputs)

    # Decision variables
    w = struct_symMX([entry('state', struct=states, repeat=form.n_mpc),
                      entry('input', struct=inputs, repeat=form.n_mpc - 1)])

    # get model constraints
    g, lbg, ubg = get_model(w, data, thetal)

    # get mpc cost function
    J = get_objective(w, data, thetal)

    # Create an NLP solver
    MPC = {"f": J, "x": w, "g": vertcat(*g), "p": data}
    options ={
    'print_time': 0,
    'ipopt' : {
    "linear_solver": "ma57",
        'print_level': 0
}}
    solverMPC = nlpsol("solver", "ipopt", MPC, options)

    return w, data, solverMPC, lbg, ubg

def set_bounds(lbw, ubw):

    lbw['state', :, 'slack'] = 0
    ubw['state', :, 'slack'] = +inf

    lbw['state', :, 'slackmin'] = 0
    ubw['state', :, 'slackmin'] = +inf

    lbw['state', 1::, 't_target'] = 10 #10
    ubw['state', 1::, 't_target'] = 31 #31

    return lbw, ubw

def set_initial_conditions(state0, lbw, ubw):
    lbw['state', 0, 'wall'] = state0['wall']
    ubw['state', 0, 'wall'] = state0['wall']

    lbw['state', 0, 'room'] = state0['room']
    ubw['state', 0, 'room'] = state0['room']

    lbw['state', 0, 't_target'] = state0['target']
    ubw['state', 0, 't_target'] = state0['target']
    #
    # lbw['state', 0, 'power'] = state0['power']
    # ubw['state', 0, 'power'] = state0['power']
    return lbw, ubw

def get_step(w, lbg, ubg, data, state0, solverMPC, spot, out_temp, t_min, t_mid, t_max, plot=True):
    # self.TimeInitial = TimeSchedule

    # get numerical data
    #forecasts = datahandling.get_mpc_data(time)
    datanum = data(0)
    datanum['t_out', :] = out_temp
    datanum['spot', :] = spot
    datanum['t_min', :] = t_min
    datanum['t_mid', :] = t_mid
    datanum['t_max', :] = t_max

    # define upper lower bound on decision variables
    ubw = w(+inf)
    lbw = w(-inf)

    # get bounds on decision variables
    lbw, ubw = set_bounds(lbw, ubw)

    # initial conditions
    w0 = w(0)
    lbw, ubw = set_initial_conditions(state0, lbw, ubw)

    # Solve NLP
    sol = solverMPC(
        x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=datanum
    )

    # fl = solverMPC.stats()
    # if not fl["success"]:
    #     raise RuntimeError("Solver infeasible")
    w_opt = sol["x"].full().flatten()

    w_opt = w(w_opt)

    mpcaction = w_opt['state', 1, 't_target'].full().flatten()[0]
    
    # ## open loop plottin
    # timesteps = list(range(288))
    # fig, (ax1, ax2) = plt.subplots(2)
    # ax1.plot(timesteps, t_max, label='t_desired')
    # ax1.plot(timesteps, t_min, label='t_min')
    # ax1.plot(timesteps, [i.full().flatten()[0] for i in w_opt['state', : , 'room']], label='t_room')
    # ax1.plot(timesteps, [i.full().flatten()[0] for i in w_opt['state', : , 't_target']], label="t_set")
    # ax1.set_xticklabels([])
    # ax1.tick_params(axis="x",
    #                 labelrotation=45,  # changes apply to the x-axis
    #                 which="both",  # both major and minor ticks are affected
    #                 bottom=False,  # ticks along the bottom edge are off
    #                 top=False,
    #                 )
    # ax1.grid()
    # handles, labels = ax1.get_legend_handles_labels()
    #
    # ax2.set_xlabel('time')
    # ax2.set_ylabel('power consumption [kW]', color='green')
    # ax2.plot(
    #     timesteps, [i.full().flatten()[0] for i in w_opt['state', : , 'power']], label="Power", color='green')
    # ax2.tick_params(axis="x", labelrotation=45)
    # ax3 = ax2.twinx()
    # ax3.set_ylabel("spot pricing", color='orange')
    # ax3.plot(
    #     timesteps, spot,
    #     label="Spot",
    #     color='orange'
    # )
    # ax3.grid()
    #
    # fig.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)  # loc='upper right')
    # plt.tight_layout()
    #
    # plt.grid("on")
    #
    # plt.show()
    # plt.close(fig)
    #
    # mpcaction =  [i.full().flatten()[0] for i in w_opt['state', 1::, 't_target']]
    return mpcaction