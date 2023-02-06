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
