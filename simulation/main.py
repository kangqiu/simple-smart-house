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
    #cacolculate non noisy power
    # COP = cfg.COP(t_out)
    COP = cfg.COP
    power = cfg.power(
        history['room'],
        t_target
    ).full().flatten()[0] # + noise_power * 0.3

    power = cfg.satpower(power).full().flatten()[0]

    t_wall = cfg.wallplus(
        history['wall'], history['room'], t_out
    ).full().flatten()[0]

    t_room = cfg.roomplus(
        history['wall'],
        history['room'],
        t_out,
        power, COP
    ).full().flatten()[0]

    t_room += noise_room
    power += noise_power * 0.3
    power = cfg.satpower(power).full().flatten()[0]


    return t_wall, t_room, power

def main():
    timesteps, timesteps_N, dt, dt_N = datahandling.get_time()

    # get outdoor temperature of the month
    # get spot pricing of the month
    spot = datahandling.get_spot_data(timesteps_N[0], timesteps_N[-1], dt_N)
    out_temp = datahandling.get_outside_temperature(timesteps_N[0], timesteps_N[-1], dt_N)
    # get minimum and desired temperature references
    t_min, t_desired = datahandling.get_temperature_settings(dt_N, cfg.start)

    # initialize history
    history = {
    'room': 17.0,
    'wall': 14.3,
    'target': 0,
    'room_noise': 0,
    'power_noise': 0,
    't_min': 17,
    't_desired': 18
}
    df_history = pd.DataFrame(columns=['room', 'wall', 'power', 'target', 'room_noise', 'power_noise', 't_min',
                                       't_desired', 'spot_price', 't_out'])

    noise  = datahandling.generate_noise_trajectories(timesteps)

    w, data, solverMPC, lbg, ubg = mpc.instantiate()

    for ts in range(len(timesteps)):
        print(f"Timestep {ts}/ {len(timesteps)}")
        spot_forecast = spot[ts:ts+cfg.n_mpc]
        out_temp_forecast = out_temp[ts:ts+cfg.n_mpc]
        t_min_reference = t_min[ts:ts+cfg.n_mpc]
        t_desired_reference = t_desired[ts:ts + cfg.n_mpc]
        t_target = mpc.get_step(w, lbg, ubg, data, history, solverMPC, spot_forecast, out_temp_forecast, t_min_reference, t_desired_reference)
        # t_target = 25
        t_wall, t_room, power = get_simulation_step(history, out_temp[ts], t_target, noise['room'][ts], noise['power'][ts])

        #append to history
        history['room'] = t_room
        history['wall'] = t_wall
        history['power'] = power
        history['target'] = t_target
        history['room_noise'] = noise['room'][ts]
        history['power_noise'] = noise['power'][ts]
        history['t_min'] = t_min[ts]
        history['t_desired'] = t_desired[ts]
        history['spot_price'] = spot[ts]
        history['t_out'] = out_temp[ts]

        df_history = pd.concat([df_history, pd.DataFrame(history, index=[0])], ignore_index=True)
    
    # some rudimentary plotting features
    datahandling.plot(df_history, 0, len(timesteps))
    print("Save results")
    f = open(cfg.results_file, "wb")
    pkl.dump(df_history, f, protocol=2)
    f.close()




########################################################################################################################

if __name__ == "__main__":
    main()

########################################################################################################################
