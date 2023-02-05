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
    history = cfg.history

    w, data, solverMPC, lbg, ubg = mpc.instantiate()

    for ts in range(len(timesteps)):

        hp = mpc.get_step(w, lbg, ubg, data, history[-1], solverMPC, ts)
        t_wall, t_room = get_simulation_step(history[-1], cfg.t_out_data[0], hp)

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
