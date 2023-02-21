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
    for k in range(cfg.n_mpc - 1):
        # COP = cfg.COP(data['t_out', k])
        COP = cfg.COP
        # power prediction
        power = cfg.power_mpc(w['state', k, 'room'], w['state', k+1, 't_target'], cfg.thetam_num)
        # Power Temperature Dynamics
        g.append(power - w['state', k, 'power'])
        lbg.append(0)
        ubg.append(0)
        # COP = cfg.COP(data['t_out', k])
        COP = cfg.COP
        t_wall_plus = cfg.wall_mpc(w['state', k, 'wall'], w['state', k, 'room'], data['t_out', k], cfg.thetam_num)
        t_room_plus = cfg.room_mpc(w['state', k, 'wall'], w['state', k, 'room'], data['t_out', k], power, COP,
                                   cfg.thetam_num)

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
        g.append(w['state', k, 't_target'] + w['input', k, 'dt_target'] - w['state', k + 1, 't_target'])
        lbg.append(0)
        ubg.append(0)

        ### Slack stuff ###
        g.append(data['t_desired', k] - w['state', k, 'room'] - w['state', k, 'slack'])
        lbg.append(-inf)
        ubg.append(0)

        g.append(data['t_min', k] - w['state', k, 'room'] - w['state', k, 'slackmin'])
        lbg.append(-inf)
        ubg.append(0)



    return g, lbg, ubg

def get_objective(w, data):
    ### Cost ###
    J = 0
    hubber = 0.5
    # weights
    w_spot = 1  # weight spot cost
    w_tbelow = 1 # weight temperature below
    w_tabove = 0.001  # weight temperature above
    w_tmin = 10
    w_target = 0.5


    for k in range(cfg.n_mpc - 1):
        J += cfg.lmpc_func(data['t_desired', k] , w['state', k, 'room'], w['state', k, 'slack'],
                           w['state', k, 'slackmin'] , w['input', k, 'dt_target'], w['state', k, 'power'],
                           data['spot', k], cfg.thetal_num)
            # J += (w_tabove * (data['t_desired', k] - w['state', k, 'room'])**2 / float(cfg.n_mpc))
            # J += (w_tbelow * w['state', k, 'slack'] ** 2 / float(cfg.n_mpc))
            # J += (w_tmin * w['state', k, 'slackmin'] ** 2 / float(cfg.n_mpc))
            # J += (w_target * (hubber ** 2) * (sqrt(1+(w['input', k, 'dt_target']/hubber) ** 2) - 1)/ float(cfg.n_mpc))
            # # J += (w_target * w['input', k, 'dt_target'] ** 2 / float(cfg.n_mpc))
            # J += (w_spot * (data['spot', k] * w['state', k, 'power']) / float(cfg.n_mpc))
    # terminal cost
    J += cfg.tmpc_func(w['state', -1, 'slack'], w['state', -1, 'slackmin'], cfg.thetal_num)
    # J += (w_tbelow * w['state', -1, 'slack'] ** 2 / float(cfg.n_mpc))
    # J += (w_tmin * w['state', -1, 'slackmin'] ** 2 / float(cfg.n_mpc))

    return J

def instantiate():
    # get variable MPC Data structure
    data = [entry('t_out', repeat=cfg.n_mpc),
            entry('spot', repeat=cfg.n_mpc),
            entry('t_min', repeat=cfg.n_mpc),
            entry('t_desired', repeat=cfg.n_mpc)]
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

def get_step(w, lbg, ubg, data, state0, solverMPC, spot, out_temp, t_min, t_desired):
    # self.TimeInitial = TimeSchedule

    # get numerical data
    #forecasts = datahandling.get_mpc_data(time)
    datanum = data(0)
    datanum['t_out', :] = out_temp
    datanum['spot', :] = spot
    datanum['t_min', :] = t_min
    datanum['t_desired', :] = t_desired

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

    mpcaction = int(np.round(w_opt['state', 1, 't_target'].full().flatten()[0]))
    ## open loop plotting?


    return mpcaction