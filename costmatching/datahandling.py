"""
data_handling.py

functions to load, preprocess, and maybe postprocess the data

"""

################################################################################################
# package imports
import sys
from casadi.tools import *
from datetime import datetime as dt
from datetime import timedelta
import scipy.interpolate as scinterp
import pandas as pd
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

################################################################################################
# file imports
import config_sim as cfg

def get_time():
    """
        creates list of timesteps based on user set start and stop time of simulation
        :return:
        """
    timestep = cfg.start
    dt = []

    # get simulator timesteps
    timesteps = [timestep]
    while timestep < (cfg.stop - timedelta(minutes=5)):
        timestep += timedelta(minutes=5)
        timesteps.append(timestep)
    for ts in timesteps:
        delta = ts - timesteps[0]
        dt.append(delta.total_seconds() / 60)

    # get simulation + prediction horizon timesteps

    timestep = cfg.start
    timesteps_N = [timestep]
    dt_N = []
    while timestep < (cfg.stop + timedelta(hours=cfg.n_mpc) - timedelta(minutes=5)):
        timestep += timedelta(minutes=5)
        timesteps_N.append(timestep)

    for ts in timesteps_N:
        delta = ts - timesteps_N[0]
        dt_N.append(delta.total_seconds() / 60)

    return timesteps, timesteps_N, dt, dt_N

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

def get_spot_data(start, stop, dt_N):
    spot_data = pkl.load(
            open(cfg.spot_file, 'rb')
        )
    data = pd.DataFrame.from_dict(spot_data)

    data['Time_start'] = data['Time_start'].dt.tz_convert(
        cfg.local_timezone
    )  # Local time zone
    mask = (data['Time_start'] >= start) & (data['Time_start'] <= stop)

    data = data.loc[mask]
    data = data.drop(columns="Time_end")

    data = data.rename(columns={"Time_start": "time", "Price": "prices"})

    prices = data.to_dict(orient="list")

    prices = interpolate_spot_data(prices, start, dt_N)

    return prices

def get_outside_temperature(start, stop, dt_N):
    met_data = pkl.load(
            open(os.path.join(cfg.temp_file), 'rb')
        )
    seklima = pd.DataFrame.from_dict(met_data)

    # convert timezones
    seklima['Time'] = seklima['Time'].dt.tz_convert(cfg.local_timezone)

    # rename and drop columns
    # seklima = seklima.drop(
    #    columns=["Duggpunktstemperatur", "Relativ luftfuktighet"]
    # )
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
        np.interp(dt_N, met_time, met_data["temperature"])
        )

    return outtemp


def generate_noise_trajectories(timesteps):

    """ generates noise trajectory """
    if cfg.add_noise:
        # check if the noise file exists
        if os.path.isfile(cfg.noise_file): # load noise file and check if the trajectories match
            with open(cfg.noise_file, 'rb') as handle:
                noise = pkl.load(handle)
            if len(noise['room']) < len(timesteps):
                raise Exception("Room temperature noise trajectory too short!")
            if len(noise['power']) <len(timesteps):
                raise Exception("Power  noise trajectory too short!")

        else: # generates new noise file
            power_noise = [0]
            room_noise = [0]
            for ts in range(len(timesteps)):
                power_noise.append(cfg.noise['beta']['power'] * power_noise[-1] + np.random.normal(cfg.noise['mu']['power'], cfg.noise['epsilon']['power'] * cfg.noise['sig']['power']))
                room_noise.append(cfg.noise['beta']['room'] * room_noise[-1] + np.random.normal(cfg.noise['mu']['room'], cfg.noise['epsilon']['room'] * cfg.noise['sig']['room']))
            #save noise in file 
            noise = {'power': power_noise[1::], 'room': room_noise[1::]}
            with open(cfg.noise_file, 'wb') as handle:
                pkl.dump(noise, handle, protocol=-1)

    else: #set noise to 0
        power_noise = [0]
        room_noise = [0]
        for ts in range(len(timesteps)):
            power_noise.append(0)
            room_noise.append(0)

        noise = {'power': power_noise[1::], 'room': room_noise[1::]}
    
    return noise

def get_temperature_settings(dt, start):
    times = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    min_temps = [17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17]
    desired_temps = [18, 18, 18, 18, 18, 19, 21.5, 21.5, 21.5, 21.5, 21.5, 21.5, 21.5, 21.5, 21.5, 21.5, 21.5, 21.5, 21.5, 21.5, 21.5, 21.5, 21.5, 20, 19]

    # interpolate temperature references
    timeInterp = []
    for k in range(len(dt)):
        timeAbs = start + timedelta(minutes=dt[k])
        timeInterp.append(np.mod(timeAbs.hour + timeAbs.minute / 60.0, 24))

    f = scinterp.interp1d(
        np.array(times),
        np.array(min_temps),
        kind='nearest')
    t_min = list(f(np.array(timeInterp)))

    f = scinterp.interp1d(
        np.array(times),
        np.array(desired_temps),
        kind="nearest")
    t_desired = list(f(np.array(timeInterp)))

    return t_min, t_desired
def plot(df_history, start, stop):
    timesteps  = list(df_history.index.values[start:stop])
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.plot(timesteps, df_history['t_desired'].values.tolist()[start:stop], label='t_desired')
    ax1.plot(timesteps, df_history['t_min'].values.tolist()[start:stop], label='t_min')
    ax1.plot(timesteps, df_history['room'].values.tolist()[start:stop], label='t_room')
    ax1.plot(timesteps, df_history['target'].values.tolist()[start:stop], label="t_set")
    ax1.set_xticklabels([])
    ax1.tick_params( axis="x", 
                labelrotation=45,  # changes apply to the x-axis
                which="both",  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                top=False,
            )
    ax1.grid()
    handles, labels = ax1.get_legend_handles_labels()

    ax2.set_xlabel('time')
    ax2.set_ylabel('power consumption [kW]', color='green')
    ax2.plot(
        timesteps, df_history['power'].values.tolist()[start:stop], label="Power", color='green')
    ax2.tick_params(axis="x", labelrotation=45)
    ax3 = ax2.twinx()
    ax3.set_ylabel("spot pricing", color='orange')
    ax3.plot(
        timesteps, df_history['spot_price'].values.tolist()[start:stop],
        label="Spot",
        color='orange'
    )
    ax3.grid()

    fig.legend(handles, labels, loc='upper center')
    plt.tight_layout()
    plt.grid("on")
    plt.show()
    plt.close(fig)
    pass


