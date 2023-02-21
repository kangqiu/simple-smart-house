import numpy as np
import pandas as pd
from gym.spaces.box import Box
import casadi as csd
import sys
sys.path.append("..")
from base_types import Env
from casadi.tools import *
import json
import datetime
from datetime import timedelta
import pickle as pkl
import pandas as pd
import scipy.interpolate as scinterp
import numpy as np
import pytz

def get_time(start_param, stop_param):
    """
        creates list of timesteps based on user set start and stop time of simulation
        :return:
        """
    local_timezone = pytz.timezone('Europe/Oslo')
    start = datetime.datetime(start_param[0], start_param[1],  start_param[2], start_param[3], start_param[4]).astimezone(local_timezone)
    stop = datetime.datetime(stop_param[0], stop_param[1], stop_param[2], stop_param[3], stop_param[4]).astimezone(local_timezone)
    dt = []
    timestep = start

    # get simulator timesteps
    timesteps = [timestep]
    while timestep < (stop - timedelta(minutes=5)):
        timestep += timedelta(minutes=5)
        timesteps.append(timestep)
    for ts in timesteps:
        delta = ts - timesteps[0]
        dt.append(delta.total_seconds() / 60)
    return timesteps, dt

def spot(start_param, stop_param, dt_sim, spot_file):
    local_timezone = pytz.timezone('Europe/Oslo')
    start = datetime.datetime(start_param[0], start_param[1], start_param[2],
                              start_param[3], start_param[4]).astimezone(local_timezone)
    stop = datetime.datetime(stop_param[0], stop_param[1], stop_param[2],
                             stop_param[3], stop_param[4]).astimezone(local_timezone)

    spot_data = pkl.load(open(spot_file, 'rb'))
    data = pd.DataFrame.from_dict(spot_data)
    mask = (data['Time_start'] >= start) & (data['Time_start'] <= stop)
    data = data.loc[mask]
    data = data.drop(columns="Time_end")
    data = data.rename(columns={"Time_start": "time", "Price": "prices"})
    prices = data.to_dict(orient="list")
    times = {}
    times['spot'] = []
    for time in prices["time"]:
        times['spot'].append((time - start).total_seconds() / 60.0)
    spotTimes = []
    spotPrices = []
    for k, price in enumerate(prices['prices']):
        spotTimes.append(times['spot'][k])
        spotTimes.append(times['spot'][k] + 60)
        spotPrices.append(price)
        spotPrices.append(price)

    f = scinterp.interp1d(
        np.array(spotTimes),
        np.array(spotPrices),
        kind='nearest',
        fill_value=spotPrices[-1],
        bounds_error=False,
    )
    spot = f(np.array(dt_sim))

    return spot

def get_temperature_settings(dt, start_param):
    local_timezone = pytz.timezone('Europe/Oslo')
    start = datetime.datetime(start_param[0], start_param[1], start_param[2], start_param[3],
                              start_param[4]).astimezone(local_timezone)
    times = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    min_temps = [17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17]
    desired_temps = [18, 18, 18, 18, 18, 19, 21.5, 21.5, 21.5, 21.5, 21.5, 21.5, 21.5, 21.5, 21.5, 21.5, 21.5, 21.5,
                     21.5, 21.5, 21.5, 21.5, 21.5, 20, 19]

    # interpolate temperature references
    timeInterp = []
    for k in range(len(dt)):
        timeAbs = start + timedelta(minutes=dt[k])
        timeInterp.append(np.mod(timeAbs.hour + timeAbs.minute / 60.0, 24))

    f = scinterp.interp1d(
        np.array(times),
        np.array(min_temps),
        kind='nearest')
    t_min = f(np.array(timeInterp))

    f = scinterp.interp1d(
        np.array(times),
        np.array(desired_temps),
        kind="nearest")
    t_desired = f(np.array(timeInterp))

    return t_min, t_desired

def get_noise(noise_file):
    with open(noise_file, 'rb') as handle:
        noise = pkl.load(handle)

    return noise

class SimpleHouse(Env):
    def __init__(self, env_params={}):
        # state box, dimension of 3, t_room, t_wall, t_set
        self.observation_space = Box(
            low=np.array([0, 0, 10, -1]),
            high=np.array([31, 31, 31, 2], dtype=np.float32),
        )
        # action box, dimension of 1, delta_tset
        self.action_space = Box(
            low=np.array([-20]),
            high=np.array([20], dtype=np.float32),
        )

        # random walking noise
        data = pkl.load(open(env_params["noise"], 'rb'))
        self.noise_hp = data["power"]
        self.noise_temp = data["room"]

        # sampling time
        self.dt = env_params["dt"]

        # simulation window
        timesteps, dt_sim = get_time(env_params["start"], env_params["stop"])

        # outdoor temperature and spot price
        # data = pd.read_excel('../../data/smart_home_uncertainties_new.ods', dtype=float)
        self.tout = env_params["t_out"] * np.ones(len(timesteps))
        # print(self.tout.shape)
        # data = pd.read_excel('../../data/smart_home_prices_new.ods', dtype=float)
        self.price = spot(env_params["start"], env_params["stop"], dt_sim, env_params["spot"])
        # desired and minimum temperature datas
        self.t_desired, self.t_min = get_temperature_settings(dt_sim, env_params["start"])

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
    def model_real(self, state, action, noise_hp, noise_t):
        # states
        t_room = state[0]
        t_wall = state[1]
        t_set = state[2]
        p_hp = state[3]

        # inputs
        delta_tset = action

        # p_hp_unsat = self.k * (t_set - t_room) + noise_hp
        # alpha = 1/self.relu * (csd.log(1 + csd.exp(self.relu * p_hp_unsat)))
        # p_hp = alpha - 1/self.relu * (csd.log(1 + csd.exp(self.relu * (alpha - self.maxpow))))

        # uncertainties
        t_out = self.tout[0]

        # Wall and Room Temperature
        t_room_next = 1 / self.m_air * (self.m_air * t_room
                                        + self.rho_in * (t_wall - t_room)
                                        + self.rho_dir * (t_out - t_room)
                                        + self.COP * p_hp) + noise_t

        t_wall_next = 1/self.m_wall * (self.m_wall * t_wall
                                       + self.rho_out * (t_out - t_wall)
                                       + self.rho_in * (t_room - t_wall))

        t_set_next = t_set + delta_tset

        p_hp_unsat = self.k * (t_set_next - t_room_next) + 0.3 * noise_hp
        p_hp_unsat_max = 1 / 2 * (p_hp_unsat + 1.5 - csd.sqrt((p_hp_unsat - 1.5) ** 2 + 0.1))
        p_hp_next = 1 / 2 * (p_hp_unsat_max + 0 + csd.sqrt((p_hp_unsat_max - 0) ** 2 + 0.1))

        next_state = np.concatenate([t_room_next, t_wall_next, t_set_next, p_hp_next], axis=None)
        return next_state

    def model_mpc(self, state: csd.SX, action: csd.SX, theta_model: csd.SX):
        # states
        t_room = state[0]
        t_wall = state[1]
        t_set = state[2]
        p_hp = state[3]

        # inputs
        delta_tset = action

        # p_hp_unsat = self.k * (t_set - t_room) + theta_model[2]
        # alpha = 1 / self.relu * (csd.log(1 + csd.exp(self.relu * p_hp_unsat)))
        # p_hp = alpha - 1 / self.relu * (csd.log(1 + csd.exp(self.relu * (alpha - self.maxpow))))

        # uncertainties
        t_out = self.tout[0]

        # Wall and Room Temperature
        t_room_next = 1 / self.m_air * (self.m_air * t_room
                                        + theta_model[0] * self.rho_in * (t_wall - t_room)
                                        + theta_model[1] * self.rho_dir * (t_out - t_room)
                                        + theta_model[2] * self.COP * p_hp) + theta_model[3]

        t_wall_next = 1 / self.m_wall * (self.m_wall * t_wall
                                         + theta_model[4] * self.rho_out * (t_out - t_wall)
                                         + theta_model[5] * self.rho_in * (t_room - t_wall)) + theta_model[6]
        t_set_next = t_set + delta_tset
        p_hp_unsat = theta_model[7] * self.k * (t_set_next - t_room_next) + theta_model[8]
        p_hp_unsat_max = 1 / 2 * (p_hp_unsat + 1.5 - csd.sqrt((p_hp_unsat - 1.5) ** 2 + 0.1))
        p_hp_next = 1 / 2 * (p_hp_unsat_max + 0 + csd.sqrt((p_hp_unsat_max - 0) ** 2 + 0.1))

        next_state = csd.vertcat(t_room_next, t_wall_next, t_set_next, p_hp_next)
        return next_state

    def reset(self):
        self.state = np.array([20, 14, 20, 0])

        self.state = self.state.clip(
            self.observation_space.low, self.observation_space.high
        )

        self.t = 0
        return self.state

    def step(self, action):
        self.state = self.model_real(
            self.state,
            action,
            self.noise_hp[self.t],
            self.noise_temp[self.t])

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
        p_hp = state[3]
        delta_tset = action

        # p_hp_unsat = self.k * (t_set - t_room)
        # alpha = 1 / self.relu * (csd.log(1 + csd.exp(self.relu * p_hp_unsat)))
        # p_hp = alpha - 1 / self.relu * (csd.log(1 + csd.exp(self.relu * (alpha - self.maxpow))))

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


