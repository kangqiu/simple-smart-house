import numpy as np
import casadi as csd
import math
from RLMPC.helpers import tqdm_context


class RLMPC_LSTDQ_Agent:
    def __init__(self, env, agent_params):
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        # Hyper-parameters
        self._parse_agent_params(**agent_params)

        # Actor initialization
        self.actor = Custom_MPCActor(self.env, self.horizon, self.cost_params, self.gamma)

        # Critic params
        self.n_sfeature = int((self.obs_dim + 2) * (self.obs_dim + 1) / 2)
        self.critic_wt = 0.01 * np.random.rand(self.n_sfeature, 1)
        self.adv_wt = 0.01 * np.random.rand(self.actor.actor_wt.shape[0], 1)
        # self.adv_wt = np.array([0, 0, 0, 1, 1, 1, 1])

        # Render prep
        self.fig = None
        self.ax = None

    # e.g., s1^2+s2^2+s1s2+s1+s2+1
    def state_to_feature(self, state):
        SS = np.triu(np.outer(state, state))
        size = state.shape[0]
        phi_s = []
        for row in range(size):
            for col in range(row, size):
                phi_s.append(SS[row][col])
        phi_s = np.concatenate((phi_s, state, 1.0), axis=None)[:, None]
        return phi_s

    # return state value function V(s)
    def get_value(self, state):
        phi_S = self.state_to_feature(state)
        V = np.matmul(phi_S.T, self.critic_wt)
        return V

    # return act, info
    def get_action(self, state, act_wt=None, time=None, mode="train"):
        pi, info = self.actor.act_forward(state, act_wt=act_wt, time=time)  # act_wt = self.actor.actor_wt
        if mode == "train":
            act = pi + self.eps * (np.random.rand(self.action_dim))  # the added noise should de changed
            act = act.clip(self.env.action_space.low, self.env.action_space.high)
            # calculate dPidP and save to info
            dpi_dtheta_s = self.actor.dPidP(state, info['act_wt'], info)
            info["dpi_dtheta_s"] = dpi_dtheta_s
        else:  # mode == "eval" or "mpc"
            act = pi
            act = act.clip(self.env.action_space.low, self.env.action_space.high)
        return act, info

    def train(self, replay_buffer, train_it):
        # Critic param update
        Av = np.zeros(shape=(self.critic_wt.shape[0], self.critic_wt.shape[0]))
        bv = np.zeros(shape=(self.critic_wt.shape[0], 1))
        for _ in tqdm_context(range(train_it), desc="Training Iterations"):
            states, actions, rewards, next_states, dones, infos = replay_buffer.sample(self.batch_size)

            for j, s in enumerate(states):
                S = self.state_to_feature(s)
                temp = (
                        S
                        - (1 - dones[j])
                        * self.gamma
                        * self.state_to_feature(next_states[j])
                )
                Av += np.matmul(S, temp.T)
                bv += rewards[j] * S
        # update self.critic_wt
        self.critic_wt = np.linalg.solve(Av, bv)

        # Advantage fn param update
        Aq = np.zeros(shape=(self.adv_wt.shape[0], self.adv_wt.shape[0]))
        bq = np.zeros(shape=(self.adv_wt.shape[0], 1))
        G = np.zeros(shape=(self.adv_wt.shape[0], self.adv_wt.shape[0]))
        for _ in tqdm_context(range(train_it), desc="Training Iterations"):
            states, actions, rewards, next_states, dones, infos = replay_buffer.sample(self.batch_size)

            for j, s in enumerate(states):
                info = infos[j]
                soln = info["soln"]
                pi_act = soln["x"].full()[: self.action_dim][:, 0]
                jacob_pi = info["dpi_dtheta_s"]
                psi = np.matmul(
                    jacob_pi, (actions[j] - pi_act)[:, None]
                )  # psi: [n_sfeature*n_a, 1]

                Aq += np.matmul(psi, psi.T)
                bq += psi * (
                        rewards[j]
                        + (1 - dones[j]) * self.gamma * self.get_value(next_states[j])
                        - self.get_value(s)
                )
                G += np.matmul(jacob_pi.T, jacob_pi)

        if np.linalg.det(Aq) != 0.0:
            # update self.adv_wt
            self.adv_wt = np.linalg.solve(Aq, bq)
            # update self.actor.actor_wt
            self.actor.actor_wt -= (self.actor_lr / self.batch_size) * np.matmul(G, self.adv_wt)
            print("params updated")

        print(self.actor.actor_wt)

    def _parse_agent_params(
            self,
            cost_params,
            eps,
            gamma,
            actor_lr,
            horizon,
            debug,
            train_params,
            constrained_updates=True,
            experience_replay=False,
    ):
        self.cost_params = cost_params
        self.eps = eps
        self.gamma = gamma
        self.actor_lr = actor_lr
        self.horizon = horizon
        self.constrained_updates = constrained_updates
        self.experience_replay = experience_replay
        self.debug = debug
        self.iterations = train_params["iterations"]
        self.batch_size = train_params["batch_size"]


class Custom_QP_formulation:
    def __init__(self, env, opt_horizon, gamma=1.0):
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.N = opt_horizon
        self.gamma = gamma
        self.etau = 1e-8

        # Symbolic variables for optimization problem
        self.x = csd.MX.sym("x", self.obs_dim)   # t_room, t_wall, t_set
        self.u = csd.MX.sym("u", self.action_dim)   # delta_tset
        self.slack = csd.MX.sym('slack', 1)
        self.slackmin = csd.MX.sym('slackmin', 1)
        self.sigma = csd.vertcat(self.slack, self.slackmin)
        self.sigma_dim = self.sigma.size()[0]
        self.X = csd.MX.sym("X", self.obs_dim, self.N)
        self.U = csd.MX.sym("U", self.action_dim, self.N)
        self.SIGMA = csd.MX.sym("SIGMA", self.sigma_dim, self.N)
        self.Opt_Vars = csd.vertcat(
            csd.reshape(self.U, -1, 1),
            csd.reshape(self.X, -1, 1),
            csd.reshape(self.SIGMA, -1, 1), )

        # Symbolic variables for all parameters: uncertainty, price, theta
        # # uncertainty dimension = N
        # self.t_out = csd.MX.sym('t_out')
        # self.T_OUT =  csd.MX.sym("T_OUT", self.N)
        # price dimension = N
        self.price = csd.MX.sym('price')
        self.PRICE = csd.MX.sym("PRICE", self.N)
        # desired temperature dimension = N
        self.t_desired = csd.MX.sym('t_desired')
        self.T_DESIRED = csd.MX.sym('T_DESIRED', self.N)
        # dt target temperature dimension = N
        self.t_min = csd.MX.sym('t_min')
        self.T_MIN = csd.MX.sym('T_MIN', self.N)
        # theta dimension = 8 + 9
        self.theta_model = csd.MX.sym("theta_model", 9)
        self.theta_temp = csd.MX.sym("theta_temp", 3)
        self.theta_spot = csd.MX.sym("theta_spot", 1)
        self.theta_act = csd.MX.sym("theta_act", 1)
        self.theta_term = csd.MX.sym("theta_term", 3)
        self.theta = csd.vertcat(self.theta_model, self.theta_temp, self.theta_spot, self.theta_act, self.theta_term)
        self.theta_dim = self.theta.size()[0]

        # [Initial state=3, theta params=7, uncertainty=N, price=N, desired temp=N, dt_target=N]
        self.P = csd.vertcat(self.x, self.theta, csd.reshape(self.PRICE, -1, 1),
                             csd.reshape(self.T_DESIRED, -1, 1), csd.reshape(self.T_MIN, -1, 1))
        # note that csd.reshape() is reshaped by column
        self.p_dim = self.obs_dim + self.theta_dim + 3 * self.N

        # cost function
        # self.stage_cost = self.stage_cost_fn()
        self.terminal_cost = self.terminal_cost_fn()
        self.spot_cost = self.spot_cost_fn()
        self.act_cost = self.act_cost_fn()
        self.temp_cost = self.temp_cost_fn()

        self.lbg_vcsd = None
        self.ubg_vcsd = None
        self.vsolver = None
        self.dPi = None

        # Optimization formulation with sensitivity (symbolic)
        self.opt_formulation()

    def opt_formulation(self):
        # Optimization cost and associated constraints
        J = 0
        W = np.array([[self.env.w_tbelow, self.env.w_tmin]])
        g = []  # Equality constraints
        hx = []  # Box constraints on states
        hsg = []  # Box constraints on sigma

        # input inequalities
        # xxxxxxxxxxxxxxxxxx

        # initial model
        xn = self.env.model_mpc(self.x, self.U[:, 0], self.theta_model)
        J += self.gamma ** 0 * (self.spot_cost(self.x, self.theta, self.PRICE[0]) + self.act_cost(self.U[:, 0]))

        for i in range(self.N - 1):
            J += self.gamma ** i * (self.temp_cost(self.X[:, i], self.theta, self.T_DESIRED[i])
                                    + W @ (self.SIGMA[:, i] * self.SIGMA[:, i] * csd.vertcat(self.theta_temp[1], self.theta_temp[2]))) \
                 + self.gamma ** (i + 1) * (self.act_cost(self.U[:, i + 1]) + self.spot_cost(self.X[:, i], self.theta, self.PRICE[i + 1])) \
                 + self.theta_act

            # model equality
            g.append(self.X[:, i] - xn)
            xn = self.env.model_mpc(self.X[:, i], self.U[:, i + 1], self.theta_model)

            # sys inequalities
            hx.append(self.T_DESIRED[i] - self.X[0, i] - self.SIGMA[0, i])  # slack
            hx.append(self.T_MIN[i] - self.X[0, i] - self.SIGMA[1, i])   # slackmin
            hx.append(10 - self.X[2, i])
            hx.append(self.X[2, i] - 31)

            # slack inequalities
            hsg.append(-self.SIGMA[0, i])
            hsg.append(-self.SIGMA[1, i])

            # input inequalities
            # xxxxxxxxxxxxxxxxxxxxxx

        J += self.gamma ** (self.N - 1) * (self.terminal_cost(self.X[:, self.N - 1], self.theta, self.T_DESIRED[self.N - 1])
                                           + W @ (self.SIGMA[:, self.N - 1] * self.SIGMA[:, self.N - 1] * csd.vertcat(self.theta_term[1], self.theta_term[2])))

        g.append(self.X[:, self.N - 1] - xn)
        hx.append(self.T_DESIRED[self.N - 1] - self.X[0, self.N - 1] - self.SIGMA[0, self.N - 1])
        hx.append(self.T_MIN[self.N - 1] - self.X[0, self.N - 1] - self.SIGMA[1, self.N - 1])
        hx.append(10 - self.X[2, self.N - 1])
        hx.append(self.X[2, self.N - 1] - 31)
        hsg.append(-self.SIGMA[0, self.N - 1])
        hsg.append(-self.SIGMA[1, self.N - 1])

        # Constraints
        G = csd.vertcat(*g)
        Hx = csd.vertcat(*hx, *hsg)
        G_vcsd = csd.vertcat(*g, *hx, *hsg)

        lbg = [0] * G.shape[0] + [-math.inf] * Hx.shape[0]
        ubg = [0] * G.shape[0] + [0] * Hx.shape[0]
        self.lbg_vcsd = csd.vertcat(*lbg)
        self.ubg_vcsd = csd.vertcat(*ubg)

        # NLP Problem for value function and policy approximation
        opts_setting = {
            "ipopt.max_iter": 1000,
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.mu_target": self.etau,
            "ipopt.mu_init": 1e-5,
            "ipopt.acceptable_tol": 1e-7,
            "ipopt.acceptable_obj_change_tol": 1e-7,

        }
        vnlp_prob = {
            "f": J,
            "x": self.Opt_Vars,
            "p": self.P,
            "g": G_vcsd,
        }
        self.vsolver = csd.nlpsol("vsolver", "ipopt", vnlp_prob, opts_setting)
        # self.dPi, self.dLagV = self.build_sensitivity(J, G, Hu, Hx)
        self.dR_sensfunc = self.build_sensitivity(J, G, Hx)

    def build_sensitivity(self, J, g, h):
        lamb = csd.MX.sym("lamb", g.shape[0])
        mu = csd.MX.sym("mu", h.shape[0])
        mult = csd.vertcat(lamb, mu)

        Lag = J + csd.transpose(lamb) @ g + csd.transpose(mu) @ h

        Lagfunc = csd.Function("Lag", [self.Opt_Vars, mult, self.P], [Lag])
        dLagfunc = Lagfunc.factory("dLagfunc", ["i0", "i1", "i2"], ["jac:o0:i0"])
        dLdw = dLagfunc(self.Opt_Vars, mult, self.P)
        Rr = csd.vertcat(csd.transpose(dLdw), g, mu * h + self.etau)
        z = csd.vertcat(self.Opt_Vars, mult)
        R_kkt = csd.Function("R_kkt", [z, self.P], [Rr])
        dR_sensfunc = R_kkt.factory("dR", ["i0", "i1"], ["jac:o0:i0", "jac:o0:i1"])

        return dR_sensfunc

    # def stage_cost_fn(self):
    #     p_hp_unsat = self.theta_spot * self.env.k * (self.x[2] - self.x[0])
    #     # alpha = 1 / self.env.relu * (csd.log(1 + csd.exp(self.env.relu * p_hp_unsat)))
    #     # p_hp = alpha - 1 / self.env.relu * (csd.log(1 + csd.exp(self.env.relu * (alpha - self.env.maxpow))))
    #     # l_spot = self.env.w_spot * (self.price * p_hp)
    #     l_spot = self.env.w_spot * (self.price * p_hp_unsat)
    #
    #     l_temp = self.theta_temp * self.env.w_tabove * (self.t_desired - self.x[0]) ** 2
    #
    #     l_act = self.theta_act * self.env.hubber ** 2 * (csd.sqrt(1 + (self.u / self.env.hubber) ** 2) - 1)
    #
    #     stage_cost = l_temp + l_spot + l_act
    #     stage_cost_fn = csd.Function("stage_cost_fn", [self.x, self.u, self.theta, self.price, self.t_desired], [stage_cost])
    #     return stage_cost_fn

    def spot_cost_fn(self):
        p_hp = self.x[3]
        # alpha = 1 / self.env.relu * (csd.log(1 + csd.exp(self.env.relu * p_hp_unsat)))
        # p_hp = alpha - 1 / self.env.relu * (csd.log(1 + csd.exp(self.env.relu * (alpha - self.env.maxpow))))
        # l_spot = self.env.w_spot * (self.price * p_hp)
        l_spot = self.theta_spot * self.env.w_spot * (self.price * p_hp)
        spot_cost_fn = csd.Function("spot_cost_fn", [self.x, self.theta, self.price], [l_spot])
        return spot_cost_fn

    def act_cost_fn(self):
        l_act = self.env.w_target * self.env.hubber ** 2 * (csd.sqrt(1 + (self.u / self.env.hubber) ** 2) - 1)
        act_cost_fn = csd.Function("act_cost_fn", [self.u], [l_act])
        return act_cost_fn

    def temp_cost_fn(self):
        l_temp = self.theta_temp[0] * self.env.w_tabove * (self.t_desired - self.x[0]) ** 2
        temp_cost_fn = csd.Function("temp_cost_fn", [self.x, self.theta, self.t_desired], [l_temp])
        return temp_cost_fn

    def terminal_cost_fn(self):
        terminal_cost = self.theta_term[0] * self.env.w_tabove * (self.t_desired - self.x[0]) ** 2
        terminal_cost_fn = csd.Function("stage_cost_fn", [self.x, self.theta, self.t_desired], [terminal_cost])
        return terminal_cost_fn


class Custom_MPCActor(Custom_QP_formulation):
    def __init__(self, env, mpc_horizon, cost_params, gamma=1.0, debug=False):
        super().__init__(env, mpc_horizon, gamma)
        self.debug = debug

        self.p_val = np.zeros((self.p_dim, 1))
        self.actor_wt = np.array([1, 1, 1, 0, 1, 1, 0, 1, 0,
                                  1, 1, 1, 1, 0, 1, 1, 1])[:, None]

        # Test run
        # _ = self.act_forward(self.env.reset())
        self.X0 = None
        self.soln = None
        self.info =None

    def act_forward(self, state, act_wt=None, time=None):
        act_wt = act_wt if act_wt is not None else self.actor_wt
        time = time if time is not None else self.env.t

        self.p_val[: self.obs_dim, 0] = state
        self.p_val[self.obs_dim:self.obs_dim + self.theta_dim, :] = act_wt

        self.p_val[self.obs_dim + self.theta_dim:self.obs_dim + self.theta_dim + self.N, :] = \
            np.reshape(self.env.price[time:time + self.N], (-1, 1), order='F')
        self.p_val[self.obs_dim + self.theta_dim + self.N:self.obs_dim + self.theta_dim + 2 * self.N, :] = \
            np.reshape(self.env.t_desired[time:time + self.N], (-1, 1), order='F')
        self.p_val[self.obs_dim + self.theta_dim + 2 * self.N:, :] = \
            np.reshape(self.env.t_min[time:time + self.N], (-1, 1), order='F')
        # order='F' reshape the matrix by column

        self.X0 = np.repeat(np.array([0, 20, 14, 20, 0, 0, 0]), self.N)

        self.soln = self.vsolver(
            x0=self.X0,
            p=self.p_val,
            lbg=self.lbg_vcsd,
            ubg=self.ubg_vcsd, )
        fl = self.vsolver.stats()
        if not fl["success"]:
            raise RuntimeError("Problem is Infeasible")

        opt_var = self.soln["x"].full()
        act = np.array(opt_var[: self.action_dim])[:, 0]

        # add time info as additional infos
        self.info = {"soln": self.soln, "time": time, "act_wt": act_wt}

        if self.debug:
            print("Soln")
            print(opt_var[: self.action_dim * self.N, :].T)
            print(opt_var[self.action_dim * self.N: self.action_dim * self.N + self.obs_dim * self.N, :].T)
            print(opt_var[self.action_dim * self.N + self.obs_dim * self.N:, :].T)
        return act, self.info

    def dPidP(self, state, act_wt, info):
        soln = info["soln"]
        x = soln["x"].full()
        lam_g = soln["lam_g"].full()
        z = np.concatenate((x, lam_g), axis=0)

        self.p_val[: self.obs_dim, 0] = state
        self.p_val[self.obs_dim:self.obs_dim + self.theta_dim, :] = act_wt

        # recall the time when calculate info
        time = info["time"]
        self.p_val[self.obs_dim + self.theta_dim:self.obs_dim + self.theta_dim + self.N, :] = \
            np.reshape(self.env.price[time:time + self.N], (-1, 1), order='F')
        self.p_val[self.obs_dim + self.theta_dim + self.N:self.obs_dim + self.theta_dim + 2 * self.N, :] = \
            np.reshape(self.env.t_desired[time:time + self.N], (-1, 1), order='F')
        self.p_val[self.obs_dim + self.theta_dim + 2 * self.N:, :] = \
            np.reshape(self.env.t_min[time:time + self.N], (-1, 1), order='F')

        [dRdz, dRdP] = self.dR_sensfunc(z, self.p_val)
        dzdP = (-np.linalg.solve(dRdz, dRdP[:, self.obs_dim:self.obs_dim+self.theta_dim])).T
        # dzdP = -(csd.inv(dRdz) @ dRdP[:, self.obs_dim:self.obs_dim + self.theta_dim])
        dpi = dzdP[:, :self.action_dim]
        return dpi

    def param_update(self, lr, dJ, act_wt):
        print("Not implemented: constrained param update")
        return act_wt
