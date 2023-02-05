################################################################################################
# package imports
from casadi.tools import *

################################################################################################
# file imports
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)

import datahandling
import config_sim as cfg


def get_model(w, data):
    g = []
    lbg = []
    ubg = []
    ### power dynamics ###
    for k in range(cfg.n_mpc - 1):
        t_wall_plus = cfg.wallplus(w['state', k, 'wall'], w['state', k, 'room'], data['t_out', k])
        t_room_plus = cfg.roomplus(w['state', k, 'wall'], w['state', k, 'room'], data['t_out', k], w['input', k, 'dhp'])

        # Room Temperature Dynamics
        g.append(t_room_plus - cfg.m_air * w['state', k+1, 'room'])
        lbg.append(0)
        ubg.append(0)

        # Wall Temperature Dynamics
        g.append(t_wall_plus - cfg.m_wall * w['state', k+1, 'wall']
        )
        lbg.append(0)
        ubg.append(0)

        ### Control dynamics ###
        g.append(w['state', k, 'hp'] + w['input', k, 'dhp'] - w['state', k+1, 'hp'])
        lbg.append(0)
        ubg.append(0)


        ### Slack stuff ###
        g.append(cfg.t_desired - w['state', k, 'room'] - w['state', k, 'slack'])
        lbg.append(-inf)
        ubg.append(0)

        g.append(cfg.t_min - w['state', k, 'room'] - w['state', k, 'slackmin'])
        lbg.append(-inf)
        ubg.append(0)

    return g, lbg, ubg

def get_objective(w, data):
    ### Cost ###
    J = 0
    for k in range(cfg.n_mpc - 1):
        J += (cfg.w_tabove * (cfg.t_desired - w['state', k, 'room'])**2 / float(cfg.n_mpc))
        J += (cfg.w_tbelow * w['state', k, 'slack'] ** 2 / float(cfg.n_mpc))
        J += (cfg.w_tmin * w['state', k, 'slackmin'] ** 2 / float(cfg.n_mpc))
        J += (cfg.w_hp * w['input', k, 'dhp'] ** 2 / float(cfg.n_mpc))
        J += (cfg.w_spot * (data['spot', k] * w['state', k, 'hp']) / float(cfg.n_mpc))

    #terminal cost
    J += (cfg.w_tabove * (cfg.t_desired - w['state', -1, 'room']) ** 2 / float(cfg.n_mpc))
    J += (cfg.w_tbelow * w['state', -1, 'slack'] ** 2 / float(cfg.n_mpc))
    J += (cfg.w_tmin * w['state', -1, 'slackmin'] ** 2 / float(cfg.n_mpc))
    J += (cfg.w_spot * (data['spot', -1] * w['state', -1, 'hp']) / float(cfg.n_mpc))

    return J

def instantiate():
    # get variable MPC Data structure
    data = [entry('t_out', repeat=cfg.n_mpc),
            entry('spot', repeat=cfg.n_mpc)]
    data = struct_symMX(data)

    #get w
    states = struct_symMX([entry('room'),
                           entry('wall'),
                           entry('slackmin'),
                           entry('slack'),
                           entry('hp')])
    # MPC inputs
    inputs = struct_symMX([entry('dhp')])
    states = struct_symMX(states)
    inputs = struct_symMX(inputs)

    # Decision variables
    w = struct_symMX([entry('state', struct=states, repeat=cfg.n_mpc),
                      entry('input', struct=inputs, repeat=cfg.n_mpc - 1)])

    # get model constraints
    g, lbg, ubg = get_model(w, data)

    # get mpc cost function
    J = get_objective(w, data)

    # Create an NLP solver
    MPC = {"f": J, "x": w, "g": vertcat(*g), "p": data}
    options = {}
    options["ipopt"] = cfg.solver_options
    solverMPC = nlpsol("solver", "ipopt", MPC, options)

    return w, data, solverMPC, lbg, ubg

def set_bounds(lbw, ubw):

    lbw['state', :, 'slack'] = 0
    ubw['state', :, 'slack'] = +inf

    lbw['state', :, 'slackmin'] = 0
    ubw['state', :, 'slackmin'] = +inf

    lbw['state', 1::, 'hp'] = 0
    ubw['state', 1::, 'hp'] = +inf

    return lbw, ubw

def set_initial_conditions(state0, lbw, ubw):
    lbw['state', 0, 'wall'] = state0['wall']
    ubw['state', 0, 'wall'] = state0['wall']

    lbw['state', 0, 'room'] = state0['room']
    ubw['state', 0, 'room'] = state0['room']

    lbw['state', 0, 'hp'] = state0['hp']
    ubw['state', 0, 'hp'] = state0['hp']
    return lbw, ubw

def get_step(w, lbg, ubg, data, state0, solverMPC, time):
    # self.TimeInitial = TimeSchedule

    # get numerical data
    #forecasts = datahandling.get_mpc_data(time)
    datanum = data(0)
    datanum['t_out', :] = cfg.t_out_data[0:288]
    datanum['spot', :] = cfg.spot_price_data[0:288]

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
    w_opt = sol["x"].full().flatten()

    w_opt = w(w_opt)

    mpcaction = w_opt['state', 1, 'hp'].full().flatten()[0]

    return mpcaction