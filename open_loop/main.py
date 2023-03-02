"""
main.py

main file for simple smart house mpc example
"""
################################################################################################
# package imports
from casadi.tools import *
import pandas as pd
import pickle as pkl
from tqdm import tqdm
from matplotlib import pyplot as plt
import random

################################################################################################
# file imports
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)




def generate_noise_trajectories(length):
    params = {'mu': {'room': 0, 'power': 0.033},
             'sig': {'room': 0.00532, 'power': 0.297},
             'beta': {'room': 0.96, 'power': 0.92},
             'epsilon': {'room': 0.75, 'power': 0.68},
             }
    """ generates noise trajectory """
    # check if the noise file exists
    power_noise = [0]
    room_noise = [0]
    for ts in range(length):
        # power_noise.append(np.random.normal(0, 0.15))
        power_noise.append(
            0.3 * (
                    params['beta']['power'] * power_noise[-1]
                    + np.random.normal(params['mu']['power'],
                    params['epsilon']['power'] *
                    params['sig']['power'])))
        # room_noise.append(np.random.normal(0, 0.9))
        room_noise.append(params['beta']['room']
                          * room_noise[-1]
                          + np.random.normal(params['mu']['room'],
                           params['epsilon']['room'] *
                           params['sig']['room']))

    noise = {
        'room' : room_noise,
        'power': power_noise
    }
    return noise

def get_simulation_step(history, t_out, t_target, noise_room, noise_power):
    #cacolculate non noisy power
    # COP = cfg.COP(t_out)
    alpha = 0.1
    maxpow = 1.5
    k = 0.2
    power = k * (t_target - history['room']) + noise_power
    a = 0.5 * (power + maxpow - sqrt((power - maxpow) ** 2 + alpha))
    power = 0.5 * (a + sqrt(a ** 2 + alpha))

    # temperature prediction model
    m_air = 31.03
    m_wall = 67.22
    rho_in = 0.36
    rho_out = 0.033
    rho_dir = 0.033

    COP = 3
    wall = (1 / m_wall) * (m_wall * history['wall']
                + rho_out * (t_out - history['wall']) + rho_in *
                            (history['room'] - history['wall']))

    room = (1 / m_air) * (m_air * history['room']
                + rho_in * (history['wall'] - history['room'])
                + rho_dir * (t_out - history['room'])
                + COP * power)

    room += noise_room

    return wall, room, power

def generate_spot_market(length):
    spot = []
    for i in range(length):
        spot.append(random.randint(10,500))

    # plt.plot(range(length), spot)
    # plt.show()
    return spot

def generate_action_sequence(length):
    #x = np.arange(start, int(np.round(stop/2)), 1)
    # x = np.arange(start, stop, 1)
    # a = np.sin(2 * np.pi * x/(288*4)) *10.5  + 20.5
    #
    # a = list(a)

    actions = []
    for i in range(length):
        actions.append(np.random.uniform(10,31))
    #round them to integers
    # actions = [int(np.round(i)) for i in a]

    # for i in range(int(np.round(stop/2)), stop):
    #     actions.append(random.randint(10,31))

    return actions

def generate_outtemp_trajectory(length):
    mu = 0
    sig = 0.2
    beta = 0.997

    tout0 = np.random.uniform(low=-7.0, high=7.0)
    dtout = [0]
    for ts in range(length):
        dtout.append(beta * dtout[-1]
                + np.random.normal(mu,sig))

    tout = [tout0 + dt for dt in dtout]

    # plt.plot(range(len(tout)), tout)
    # plt.show()
    return tout

def generate_settings(length):
    tmin = []
    tmax = []
    tmid = []
    for ts in range(length):
        tmin.append(np.random.uniform(low=14.0, high=20.0))
        tmax.append(np.random.uniform(low=20.0, high=26.0))
        tmid.append((tmin[-1] + tmax[-1])/2)

    return tmin, tmid, tmax

def plot(df_history, start, stop):
    timesteps  = list(df_history.index.values[start:stop])

    plt.subplot(2,1,1)
    plt.plot(timesteps, df_history['room'].values.tolist()[start:stop], label='t_room')
    plt.plot(timesteps, df_history['target'].values.tolist()[start:stop], label="t_set")
    plt.grid()
    plt.legend()

    plt.subplot(2,1,2)
    plt.plot(
        timesteps, df_history['power'].values.tolist()[start:stop], label="Power", color='green')
    plt.grid()

    plt.legend() # loc='upper right')
    plt.tight_layout()
    plt.show()

########################################################################################################################

results_file = './results/random_runs_4.pkl'
# noise_file = './noise/noise.pkl'# 1 month simulation time frame

n_mpc = 288  # 24-hour prediction window
start = 0

n_sim = 300 # outside temperature set, but can vary

history_runs = {}

# load outdoor temperature
#
for k, index in enumerate(tqdm(range(n_sim))):

    df_history = pd.DataFrame(columns=['room', 'wall', 'power', 'room_noise', 'power_noise', 'tout', 'tmin', 'tmid',
                                       'tmax', 'spot'])
    out_temp = generate_outtemp_trajectory(n_mpc)
    tmin, tmid, tmax = generate_settings(n_mpc)
    noise = generate_noise_trajectories(n_mpc)
    spot = generate_spot_market(n_mpc)

    actions = generate_action_sequence(n_mpc)

    # initialize states randomly
    history = {
        'room': np.random.uniform(low=15, high=26),
        'wall': np.random.uniform(low=10, high=18),
        'room_noise': 0,
    }
    for ts in range(n_mpc):
        action = actions[ts]
        actionsim = int(np.round(action))
        t_wall, t_room, power = get_simulation_step(history, out_temp[ts], actionsim, noise['room'][ts], noise['power'][ts])
        history['power'] = power
        history['power_noise'] = noise['power'][ts]
        history['tout'] = out_temp[ts]
        history['tmin'] = tmin[ts]
        history['tmid'] = tmid[ts]
        history['tmax'] = tmax[ts]
        history['spot'] = spot[ts]
        history['target'] = action
        df_history = pd.concat([df_history, pd.DataFrame(history, index=[0])], ignore_index=True)

        history = {}
        #append to history
        history['room'] = t_room
        history['wall'] = t_wall
        history['room_noise'] = noise['room'][ts]
        history['tout'] = out_temp[ts]

    history_runs[k] = df_history
# plotting of state trajectories
# plot(df_history, start, stop)
print("Save results")

f = open(results_file, "wb")
pkl.dump(history_runs, f, protocol=2)
f.close()

print("Results saved")




########################################################################################################################
