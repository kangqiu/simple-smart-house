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
from datetime import datetime
import pytz

################################################################################################
# file imports
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)

import datahandling
import mpc
import formulation as form


def get_simulation_step(history, t_out, t_target, noise_room, noise_power):
    #cacolculate non noisy power
    power = form.power_func(
        history['room'],
        t_target
    ).full().flatten()[0]
    power = form.satpower_func(power).full().flatten()[0]

    t_wall = form.wall_func(
        history['wall'], history['room'], t_out
    ).full().flatten()[0]

    t_room = form.room_func(
        history['wall'],
        history['room'],
        t_out,
        power
    ).full().flatten()[0]

    # recalculate noisy power
    power = form.power_func(
        history['room'],
        t_target
    ).full().flatten()[0] + noise_power
    power = form.satpower_func(power).full().flatten()[0]


    t_room += noise_room

    return t_wall, t_room, power

def main():

    # set start and stop of simulation timeframe
    start = datetime(2022, 9, 1, 0, 0).astimezone(pytz.timezone('Europe/Oslo'))
    stop = datetime(2022, 9, 8, 0, 0).astimezone(pytz.timezone('Europe/Oslo'))

    # results file
    results_file = './results/01_closedloop_week.pkl'
    print(results_file)
    noise_file = './data/noise/September_week.pkl'
    #theta
    thetal = np.array([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    # thetal = np.array([6.17219637e-04, 1.11016402e+00, 9.99789794e-01, 9.75831310e-01,
    #  9.84418325e-01, 1.00000008e+00 ,2.15059108e-06 ,1.15412204e-01,
    #  7.41456376e-02])
    print(thetal)

    #get timesteps, as mpc has lookahead model, we need to read data to accomodate the info the controller needs
    timesteps, timesteps_N, dt, dt_N = datahandling.get_time(start, stop)
    # get outdoor temperature of the month
    # get spot pricing of the month
    spot = datahandling.get_spot_data(timesteps_N[0], timesteps_N[-1], dt_N)
    out_temp = datahandling.get_outside_temperature(timesteps_N[0], timesteps_N[-1], dt_N)
    # get minimum and desired temperature references
    t_min, t_max, t_mid = datahandling.get_temperature_settings(dt_N, start)

    # initialize history
    history = {
    'room': 17,
    'wall': 14.3,
    'target': 20,
    'power': 0,
    'room_noise': 0,
    'power_noise': 0,
    't_min': 17,
    't_desired': 18,
}
    df_history = pd.DataFrame(columns=['timestep', 'room', 'wall', 'power', 'target', 'room_noise', 'power_noise', 't_min',
                                       't_max', 't_mid', 'spot_price', 't_out'])
    noise  = datahandling.generate_noise_trajectories(timesteps, noise_file)

    w, data, solverMPC, lbg, ubg = mpc.instantiate(thetal)
    for ts, index in enumerate(tqdm(range(len(timesteps)))):
    # for ts in range(len(timesteps)):
        # print(f"Timestep {ts}/ {len(timesteps)}")
        spot_forecast = spot[ts:ts+form.n_mpc]
        out_temp_forecast = out_temp[ts:ts+form.n_mpc]
        t_min_reference = t_min[ts:ts+form.n_mpc]
        t_mid_reference = t_mid[ts:ts+form.n_mpc]
        t_max_reference = t_max[ts:ts+form.n_mpc]
        # t_target = 15
        t_target = mpc.get_step(w, lbg, ubg, data, history, solverMPC, spot_forecast, out_temp_forecast, t_min_reference,
                                t_mid_reference, t_max_reference)
        t_wall, t_room, power = get_simulation_step(history, out_temp[ts], np.round(t_target), noise['room'][ts], noise['power'][ts])

        #append same step power simulation to datafrema
        history['power'] = power
        history['target'] = t_target
        history['t_min'] = t_min[ts]
        history['t_mid'] = t_mid[ts]
        history['t_max'] = t_max[ts]
        history['spot_price'] = spot[ts]
        history['t_out'] = out_temp[ts]
        history['timestep'] = timesteps[ts]
        df_history = pd.concat([df_history, pd.DataFrame(history, index=[0])], ignore_index=True)

        #append to history state predictions
        history['room'] = t_room
        history['wall'] = t_wall
        history['room_noise'] = noise['room'][ts]
        history['power_noise'] = noise['power'][ts]


    # append last timestep to dataframe
    history['power'] = 0
    history['target'] = 0
    history['t_min'] = t_min[ts+1]
    history['t_mid'] = t_mid[ts+1]
    history['t_max'] = t_max[ts+1]
    history['spot_price'] = spot[ts+1]
    history['t_out'] = out_temp[ts+1]

    df_history = pd.concat([df_history, pd.DataFrame(history, index=[0])], ignore_index=True)

    # some rudimentary plotting features
    datahandling.plot(df_history, 0, len(timesteps))
    print("Save results")


    f = open(results_file, "wb")
    pkl.dump(df_history, f, protocol=2)
    f.close()

    print("Results saved")



########################################################################################################################

if __name__ == "__main__":
    main()

########################################################################################################################
