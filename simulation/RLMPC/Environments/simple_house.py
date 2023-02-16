import numpy as np
import pandas as pd
from gym.spaces.box import Box
import casadi as csd
from RLMPC.base_types import Env
from casadi.tools import *
import json


class SimpleHouse(Env):
    def __init__(self, env_params={}):
        # state box, dimension of 3, t_room, t_wall, t_set
        self.observation_space = Box(
            low=np.array([0, 0, 10]),
            high=np.array([31, 31, 31], dtype=np.float32),
        )
        # action box, dimension of 1, delta_tset
        self.action_space = Box(
            low=np.array([-20]),
            high=np.array([20], dtype=np.float32),
        )

        # random walking noise
        self.epsilon_hp = env_params["epsilon_hp"]
        self.epsilon_temp = env_params["epsilon_temp"]

        # sampling time
        self.dt = env_params["dt"]

        # outdoor temperature and spot price
        data = pd.read_excel('../data/smart_home_uncertainties_new.ods', dtype=float)
        self.tout = data.t_out.to_numpy()
        # print(self.tout.shape)
        data = pd.read_excel('../data/smart_home_prices_new.ods', dtype=float)
        self.price = data.price_buy.to_numpy()
        # desired and minimum temperature datas
        self.t_desired = 18 * np.ones(1000)
        self.t_min = 14 * np.ones(1000)

        # time
        self.t = None

        # system parameters
        self.m_air = 31.02679204362912
        self.m_wall = 67.21826736655125
        self.rho_in = 0.36409940390361406
        self.rho_out = 0.03348756113438382
        self.rho_dir = 0.03348756179891388
        self.COP = 3
        self.k = 0.2
        self.relu = 100
        self.maxpow = 1.5

        # weights parameters
        self.w_spot = 0.1  # weight spot cost
        self.w_tbelow = 0.2  # weight temperature below
        self.w_tabove = 0.005  # weight temperature above
        self.w_tmin = 50
        self.w_target = 0.5
        self.hubber = 0.5

        self.state = None
        # initialization
        self.reset()

    # dimension=2, 1, 1
    def model_real(self, state, action, uncertainty):
        # states
        t_room = state[0]
        t_wall = state[1]
        t_set = state[2]

        # inputs
        delta_tset = action

        p_hp_unsat = self.k * (t_set - t_room) + np.random.normal(0, self.epsilon_hp)
        alpha = 1/self.relu * (csd.log(1 + csd.exp(self.relu * p_hp_unsat)))
        p_hp = alpha - 1/self.relu * (csd.log(1 + csd.exp(self.relu * (alpha - self.maxpow))))

        # uncertainties
        t_out = uncertainty

        # Wall and Room Temperature
        t_room_next = 1 / self.m_air * (self.m_air * t_room
                                        + self.rho_in * (t_wall - t_room)
                                        + self.rho_dir * (t_out - t_room)
                                        + self.COP * p_hp) + np.random.normal(0, self.epsilon_temp)

        t_wall_next = 1/self.m_wall * (self.m_wall * t_wall
                                       + self.rho_out * (t_out - t_wall)
                                       + self.rho_in * (t_room - t_wall)) + np.random.normal(0, self.epsilon_temp)

        t_set_next = t_set + delta_tset

        next_state = np.concatenate([t_room_next, t_wall_next, t_set_next], axis=None)
        return next_state

    def model_mpc(self, state: csd.SX, action: csd.SX, uncertainty: csd.SX, theta_model: csd.SX):
        # states
        t_room = state[0]
        t_wall = state[1]
        t_set = state[2]

        # inputs
        delta_tset = action

        p_hp_unsat = self.k * (t_set - t_room) + theta_model[2]
        alpha = 1 / self.relu * (csd.log(1 + csd.exp(self.relu * p_hp_unsat)))
        p_hp = alpha - 1 / self.relu * (csd.log(1 + csd.exp(self.relu * (alpha - self.maxpow))))

        # uncertainties
        t_out = uncertainty

        # Wall and Room Temperature
        t_room_next = 1 / self.m_air * (self.m_air * t_room
                                        + self.rho_in * (t_wall - t_room)
                                        + self.rho_dir * (t_out - t_room)
                                        + self.COP * p_hp) + theta_model[0]

        t_wall_next = 1 / self.m_wall * (self.m_wall * t_wall
                                         + self.rho_out * (t_out - t_wall)
                                         + self.rho_in * (t_room - t_wall)) + theta_model[1]
        t_set_next = t_set + delta_tset
        next_state = csd.vertcat(t_room_next, t_wall_next, t_set_next)
        return next_state

    def reset(self):
        self.state = np.array([14.5, 10, 18])

        self.state = self.state.clip(
            self.observation_space.low, self.observation_space.high
        )

        self.t = 0
        return self.state

    def step(self, action):
        self.state = self.model_real(
            self.state,
            action,
            self.tout[self.t])

        # self.state = self.state.clip(
        #     self.observation_space.low, self.observation_space.high
        # )

        rew, done = self.reward_fn(self.state, action, self.price[self.t], self.t_desired[self.t])

        self.t += 1
        return self.state, rew, done

    # return r, done
    def reward_fn(self, state, action, price, t_desired):
        t_room = state[0]
        t_wall = state[1]
        t_set = state[2]
        delta_tset = action

        p_hp_unsat = self.k * (t_set - t_room)
        alpha = 1 / self.relu * (csd.log(1 + csd.exp(self.relu * p_hp_unsat)))
        p_hp = alpha - 1 / self.relu * (csd.log(1 + csd.exp(self.relu * (alpha - self.maxpow))))

        l_temp = self.w_tabove * (t_desired - t_room) ** 2
        l_spot = self.w_spot * (price * p_hp)
        r = l_spot + l_temp

        done = 0
        return r, done

    # not in use
    def cost_fn(self, state, action, next_state):
        pass


# test
if __name__ == "__main__":
    print(os.getcwd())
    with open('../Settings/rl_mpc_lstd.json', 'r') as f:
        params = json.load(f)
        print(params)
    a = SimpleHouse(params["env_params"])
    print(a.state)
    print(a.step(0.5))
    a.reset()
    print(a.state)
    a.step(0.1)
    a.step(0.1)
    print(a.t)


