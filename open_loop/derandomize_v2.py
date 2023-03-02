"""
open loop simulations

derandomized spot market and outdoor temperature
derandomized tmax, tmin, tmid trajectories

states get initialized randomly -> room between 15 and 23.5 (which is +-2 degrees outside of comfort zone)

random action sequence, RW noise trajectory
"""
################################################################################################
# package imports
from casadi.tools import *
import pandas as pd
import pickle as pkl
from tqdm import tqdm
from matplotlib import pyplot as plt
import random
import pytz
import datetime
from dateutil.relativedelta import relativedelta
import scipy.interpolate as scinterp
################################################################################################
# file imports
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)

def get_time(start, stop):
    """
        creates list of timesteps based on user set start and stop time of simulation
        :return:
        """
    timestep = start
    dt = []
    # get simulator timesteps
    timesteps = [timestep]
    while timestep < (stop - datetime.timedelta(minutes=5)):
        timestep += datetime.timedelta(minutes=5)
        timesteps.append(timestep)
    for ts in timesteps:
        delta = ts - timesteps[0]
        dt.append(delta.total_seconds() / 60)

    return timesteps, dt


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
    power = k * (t_target - history['room'])
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

    #recalculate noisy power
    power = k * (t_target - history['room']) + noise_power
    a = 0.5 * (power + maxpow - sqrt((power - maxpow) ** 2 + alpha))
    power = 0.5 * (a + sqrt(a ** 2 + alpha))

    return wall, room, power

def interpolate_spot_data(spot, start, dt):
    times = {}
    times['spot'] = []
    for time in spot["time"]:
        times['spot'].append((time - start).total_seconds() / 60.0)
    spotTimes = []
    spotPrices = []
    for k, price in enumerate(spot['prices']):
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
    spot = list(f(np.array(dt)))
    return spot

def generate_spot_market(start, stop, dt):
    spot_data = pkl.load(
            open('../data/SpotData2022_Trheim.pkl', 'rb')
        )
    data = pd.DataFrame.from_dict(spot_data)

    data['Time_start'] = data['Time_start'].dt.tz_convert(
       pytz.timezone('Europe/Oslo')
    )  # Local time zone
    mask = (data['Time_start'] >= start) & (data['Time_start'] <= stop)
    data = data.loc[mask]
    data = data.drop(columns="Time_end")
    data = data.rename(columns={"Time_start": "time", "Price": "prices"})
    prices = data.to_dict(orient="list")
    prices = interpolate_spot_data(prices, start, dt)
    return prices


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

def generate_outtemp_trajectory(start, stop, dt):

    start = start - relativedelta(years=1)
    stop = stop - relativedelta(years=1)
    met_data = pkl.load(
            open(os.path.join('../data/SEKLIMAData_2021.pkl'), 'rb')
        )
    seklima = pd.DataFrame.from_dict(met_data)

    # convert timezones
    local_timezone =  pytz.timezone('Europe/Oslo')
    seklima['Time'] = seklima['Time'].dt.tz_convert(local_timezone)
    seklima = seklima.rename(columns={"Lufttemperatur": "temperature"})
    mask = (seklima["Time"] >= start) & (seklima["Time"] <= stop)
    data = seklima.loc[mask]
    met_data = data.to_dict(orient="list")

    # Interpolate outside temperature for simulation
    met_time = []
    for ts in met_data["Time"]:
        delta = ts - met_data["Time"][0]
        met_time.append(delta.total_seconds() / 60)
    outtemp = list(
        np.interp(dt, met_time, met_data["temperature"])
        )
    return outtemp

def generate_settings(dt, start):
    times = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    min_temps = [17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17]
    max_temps = [18, 18, 18, 18, 18, 19, 21.5, 21.5, 21.5, 21.5, 21.5, 21.5, 21.5, 21.5, 21.5, 21.5, 21.5, 21.5, 21.5,
                 21.5, 21.5, 21.5, 21.5, 20, 19]
    mid_temps = [17.5, 17.5, 17.5, 17.5, 17.5, 18, 19.25, 19.25, 19.25, 19.25, 19.25, 19.25, 19.25, 19.25, 19.25, 19.25,
                 19.25, 19.25, 19.25, 19.25, 19.25, 19.25, 19.25, 18.5, 18]
    # interpolate temperature references
    timeInterp = []
    for k in range(len(dt)):
        timeAbs = start + datetime.timedelta(minutes=dt[k])
        timeInterp.append(np.mod(timeAbs.hour + timeAbs.minute / 60.0, 24))

    f = scinterp.interp1d(
        np.array(times),
        np.array(min_temps),
        kind='nearest')
    t_min = list(f(np.array(timeInterp)))

    f = scinterp.interp1d(
        np.array(times),
        np.array(max_temps),
        kind="nearest")
    t_max = list(f(np.array(timeInterp)))

    f = scinterp.interp1d(
        np.array(times),
        np.array(mid_temps),
        kind="nearest")
    t_mid = list(f(np.array(timeInterp)))

    return t_min, t_mid, t_max

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

results_file = './results/derandomizev2_run1.pkl'
# noise_file = './noise/noise.pkl'# 1 month simulation time frame
start = datetime.datetime(2022, 9, 1, 0, 0).astimezone(pytz.timezone('Europe/Oslo'))
stop = datetime.datetime(2022, 10, 2, 0, 0).astimezone(pytz.timezone('Europe/Oslo'))

n_mpc = 288  # 24-hour prediction window
timesteps, dt = get_time(start, stop)

n_sim = 30000# outside temperature set, but can vary

history_runs = {}
outdoor_temperature = generate_outtemp_trajectory(start, stop, dt)
spot_pricing = generate_spot_market(start, stop, dt)
tmin_trajectory, tmid_trajectory, tmax_trajectory = generate_settings(dt[0:n_mpc*2], start)

# load outdoor temperature
#
for k, index in enumerate(tqdm(range(n_sim))):
    random_sampling = numpy.random.randint(0, len(timesteps) - n_mpc)
    random_setting = numpy.random.randint(0, n_mpc)
    df_history = pd.DataFrame(columns=['room', 'wall', 'power', 'room_noise', 'power_noise', 'tout', 'tmin', 'tmid',
                                       'tmax', 'spot'])
    spot = spot_pricing[random_sampling:random_sampling + n_mpc]
    tmin = tmin_trajectory[random_setting:random_setting + n_mpc]
    tmid = tmid_trajectory[random_setting:random_setting + n_mpc]
    tmax = tmax_trajectory[random_setting:random_setting + n_mpc]
    noise = generate_noise_trajectories(n_mpc)
    out_temp = outdoor_temperature[random_setting:random_setting +n_mpc]
    actions = generate_action_sequence(n_mpc)

    # initialize states randomly
    history = {
        'room': np.random.uniform(low=15, high=23),
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
