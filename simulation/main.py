"""
main.py

main file for simple smart house mpc example
"""
################################################################################################
# package imports
from casadi.tools import *
import pandas as pd
import pickle as pkl

################################################################################################
# file imports
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)

import datahandling
import config_sim as cfg
import mpc

def get_simulation_step(history, t_out, t_target, noise_room, noise_power):
    #calculate non noisy power
    power = cfg.power(
        history["room"],
        t_target
    ).full().flatten()[0]

    power = cfg.satpower(power).full().flatten()[0]
    t_wall = cfg.wallplus(
        history["wall"], history["room"], t_out
    ).full().flatten()[0]

    t_room = cfg.roomplus(
        history["wall"],
        history["room"],
        t_out,
        power
    ).full().flatten()[0]

    t_room += noise_room
    # compute noisy power
    power = cfg.power(history["room"], t_target).full().flatten()[0] + noise_power * 0.3
    power = cfg.satpower(power).full().flatten()[0]
    return t_wall, t_room, power

def main():
    # initialize history
    timesteps, timesteps_N, dt, dt_N = datahandling.get_time()

    # get outdoor temperature of the month
    # get spot pricing of the month
    spot = datahandling.get_spot_data(timesteps_N[0], timesteps_N[-1], dt_N)
    out_temp = datahandling.get_outside_temperature(timesteps_N[0], timesteps_N[-1], dt_N)
    # get minimum and desired temperature references
    t_min, t_desired = datahandling.get_temperature_settings(dt_N, cfg.start)


    history = cfg.history
    df_history = pd.DataFrame(columns=['room', 'wall', 'power', 'target', 'room_noise', 'power_noise', 't_min',
                                       't_desired', 'spot_price', 't_out'])

    noise  = datahandling.generate_noise_trajectories(timesteps)

    w, data, solverMPC, lbg, ubg = mpc.instantiate()

    for ts in range(len(timesteps)):
        spot_forecast = spot[ts:ts+cfg.n_mpc]
        out_temp_forecast = out_temp[ts:ts+cfg.n_mpc]
        t_min_reference = t_min[ts:ts+cfg.n_mpc]
        t_desired_reference = t_desired[ts:ts + cfg.n_mpc]
        t_target = mpc.get_step(w, lbg, ubg, data, history[-1], solverMPC, spot_forecast, out_temp_forecast, t_min_reference, t_desired_reference)
        # t_target = 25
        t_wall, t_room, power = get_simulation_step(history[-1], out_temp[ts], t_target, noise['room'][ts], noise['power'][ts])

        #append to history
        history.append({})
        history[-1]['room'] = t_room
        history[-1]['wall'] = t_wall
        history[-1]['power'] = power
        history[-1]['target'] = t_target
        history[-1]['room_noise'] = noise['room'][ts]
        history[-1]['power_noise'] = noise['power'][ts]
        history[-1]['t_min'] = t_min[ts]
        history[-1]['t_desired'] = t_desired[ts]
        history[-1]['spot_price'] = spot[ts]
        history[-1]['t_out'] = out_temp[ts]

        df_history = pd.concat([df_history, pd.DataFrame(history[-1], index=[0])], ignore_index=True)
    
    # some rudimentary plotting features
    datahandling.plot(df_history, 0, len(timesteps), spot)
    print("Save results")
    f = open(cfg.results_file, "wb")
    pkl.dump(df_history, f, protocol=2)
    f.close()




########################################################################################################################

if __name__ == "__main__":
    main()

########################################################################################################################
