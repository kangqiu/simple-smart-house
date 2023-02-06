"""
main.py

main file for simple smart house mpc example
"""
################################################################################################
# package imports
from casadi.tools import *


################################################################################################
# file imports
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)

import datahandling
import config_sim as cfg
import mpc

def get_simulation_step(history, t_out, hp):
    t_wall = cfg.wallplus(
        history["wall"], t_out, history["room"]
    ).full().flatten()[0] / cfg.m_wall
    t_room = cfg.roomplus(
        history["room"],
        history["wall"],
        t_out,
        hp,
    ).full().flatten()[0] / cfg.m_air

    return t_wall, t_room

def main():
    # initialize history
    timesteps, timesteps_N, dt, dt_N = datahandling.get_time()

    # get outdoor temperature of the month
    # get spot pricing of the month
    spot = datahandling.get_spot_data(timesteps_N[0], timesteps_N[-1], dt_N)
    out_temp = datahandling.get_outside_temperature(timesteps_N[0], timesteps_N[-1], dt_N)

    history = cfg.history

    w, data, solverMPC, lbg, ubg = mpc.instantiate()

    for ts in range(len(timesteps)):
        spot_forecast = spot[ts:ts+cfg.n_mpc]
        out_temp_forecast = out_temp[ts:ts+cfg.n_mpc]
        hp = mpc.get_step(w, lbg, ubg, data, history[-1], solverMPC, spot_forecast, out_temp_forecast)
        t_wall, t_room = get_simulation_step(history[-1], out_temp[ts], hp)

        #append to history
        history.append({})
        history[-1]['wall'] = t_wall
        history[-1]['room'] = t_room
        history[-1]['hp'] = hp


    return history


########################################################################################################################

if __name__ == "__main__":
    main()

########################################################################################################################
