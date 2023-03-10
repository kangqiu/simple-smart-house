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
from dateutil.relativedelta import relativedelta
import pytz
################################################################################################
# file imports
# import config_sim as cfg
import formulation as form

def get_time(start, stop):
    """
        creates list of timesteps based on user set start and stop time of simulation
        :return:
        """
    timestep = start
    dt = []

    # get simulator timesteps
    timesteps = [timestep]
    while timestep < (stop - timedelta(minutes=5)):
        timestep += timedelta(minutes=5)
        timesteps.append(timestep)
    for ts in timesteps:
        delta = ts - timesteps[0]
        dt.append(delta.total_seconds() / 60)

    # get simulation + prediction horizon timesteps

    timestep = start
    timesteps_N = [timestep]
    dt_N = []
    while timestep < (stop + timedelta(hours=form.n_mpc) - timedelta(minutes=5)):
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
    spot_file = './data/SpotData2022_Trheim.pkl'
    spot_data = pkl.load(
            open(spot_file, 'rb')
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
    # p =[15] * 3 +[200] *int(len(data['prices'])-3)
    # prices['prices'] = p
    prices = interpolate_spot_data(prices, start, dt_N)


    return prices

def get_outside_temperature(start, stop, dt_N):
    temp_file = './data/SEKLIMAData_2021.pkl'
    start = start - relativedelta(years=1)
    stop = stop - relativedelta(years=1)
    met_data = pkl.load(
            open(os.path.join(temp_file), 'rb')
        )
    seklima = pd.DataFrame.from_dict(met_data)

    # convert timezones
    seklima['Time'] = seklima['Time'].dt.tz_convert(pytz.timezone('Europe/Oslo'))

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

    # outtemp = [cfg.set_t_out] * len(dt_N)

    return outtemp


def generate_noise_trajectories(timesteps, noise_file):

    """ generates noise trajectory """

    # check if the noise file exists
    if os.path.isfile(noise_file): # load noise file and check if the trajectories match
        with open(noise_file, 'rb') as handle:
            noise = pkl.load(handle)
        if len(noise['room']) < len(timesteps):
            raise Exception("Room temperature noise trajectory too short!")
        if len(noise['power']) <len(timesteps):
            raise Exception("Power  noise trajectory too short!")

    else: # generates new noise file
        power_noise = [0]
        room_noise = [0]
        for ts in range(len(timesteps)):
            # power_noise.append(np.random.normal(0, 0.15))
            power_noise.append(0.3 *(form.noise['beta']['power'] * power_noise[-1]
                                     + np.random.normal(form.noise['mu']['power'],
                                                        form.noise['epsilon']['power'] * form.noise['sig']['power'])))
            room_noise.append(form.noise['beta']['room'] * room_noise[-1]
                              + np.random.normal(form.noise['mu']['room'], form.noise['epsilon']['room'] * form.noise['sig']['room']))
        plt.plot(range(len(room_noise)), room_noise)
        plt.show()
        plt.plot(range(len(power_noise)), power_noise)
        plt.show()
        #save noise in file
        noise = {'power': power_noise[1::], 'room': room_noise[1::]}
        with open(noise_file, 'wb') as handle:
            pkl.dump(noise, handle, protocol=-1)
    
    return noise

def get_temperature_settings(dt, start):
    times = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    min_temps = [17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17]
    max_temps = [18, 18, 18, 18, 18, 19, 21.5, 21.5, 21.5, 21.5, 21.5, 21.5, 21.5, 21.5, 21.5, 21.5, 21.5, 21.5, 21.5, 21.5, 21.5, 21.5, 21.5, 20, 19]
    mid_temps = [17.5, 17.5, 17.5, 17.5, 17.5, 18, 19.25, 19.25, 19.25, 19.25, 19.25, 19.25, 19.25, 19.25, 19.25, 19.25,
                 19.25, 19.25, 19.25, 19.25, 19.25, 19.25, 19.25, 18.5, 18]
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
        np.array(max_temps),
        kind="nearest")
    t_max = list(f(np.array(timeInterp)))

    f = scinterp.interp1d(
        np.array(times),
        np.array(mid_temps),
        kind="nearest")
    t_mid = list(f(np.array(timeInterp)))

    return t_min, t_max, t_mid
def plot(df_history, start, stop, savefig = False):
    target_continuous = df_history['target'].values.tolist()
    target = [np.round(t) for t in target_continuous]
    timesteps  = list(df_history.index.values[start:stop])
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.plot(timesteps, df_history['t_max'].values.tolist()[start:stop], label='t_max')
    # ax1.plot(timesteps, df_history['t_mid'].values.tolist()[start:stop], label='t_mid')
    ax1.plot(timesteps, df_history['t_min'].values.tolist()[start:stop], label='t_min')
    ax1.plot(timesteps, df_history['room'].values.tolist()[start:stop], label='t_room')
    # ax1.plot(timesteps, target[start:stop], label="t_set")
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

    # fig.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) # loc='upper right')
    fig.legend(handles, labels)
    plt.tight_layout()

    plt.grid("on")

    if savefig == True:
        filename = cfg.results_file.split('.')[1]
        fig.savefig('fig.pdf',format="pdf",  bbox_inches='tight')
    plt.show()
    plt.close(fig)
    pass


