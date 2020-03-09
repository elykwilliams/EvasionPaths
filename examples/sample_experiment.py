# Kyle Williams 3/5/20

import os
from sys import path

path.insert(1, '../src')

from evasion_path import *
from joblib import Parallel, delayed

# Setup
#   Set parameters 
#       num_sensors:        
#               Number of interior sensors
#       sensing_radius:     
#               sensing radius of the sensors
#       timestep_size:
#               maximum timestep size used in simulaiton
#       output_dir:
#               relative path to output directory, will create directory if it does not exist
#       filename_base:
#               datafiles will be titled filename_base.txt and located at output_dir/filename_base.txt
#       n_runs:
#               number of simulations to run. Will be run in parallel
#
#   Motion Model or Boundary type are set in simulate(), note that EvasionPathSimulation must be created in
#   simulation(), otherwise the same object will be used in parallel.

# Run
#   python3 sample_experiment.py &
#
#   Note, simulations may halt if an error is raised, this should be caught and handled in simulate()

# Output
#   There will be one output file in the selected output directory named filename_base.txt
#   Each line will contain an element will contain a runtime from a simulation. 

###########################
#       Parameters
###########################

num_sensors: int = 20
sensing_radius: float = 0.2
timestep_size: float = 0.01

output_dir: str = "./output"
filename_base: str = "data4"

n_runs: int = 1


def simulate() -> float:
    unit_square = Boundary(spacing=sensing_radius)

    brownian_motion = BrownianMotion(dt=timestep_size,
                                     sigma=0.01,
                                     sensing_radius=sensing_radius,
                                     boundary=unit_square)

    simulation = EvasionPathSimulation(boundary=unit_square,
                                       motion_model=brownian_motion,
                                       n_sensors=num_sensors,
                                       sensing_radius=sensing_radius,
                                       dt=timestep_size)

    return simulation.run()


def output_data(filename: str, data_points: list) -> None:
    with open(filename, 'a+') as file:
        file.writelines("%.2f\n" % d for d in data_points)


def run_experiment() -> None:
    times = Parallel(n_jobs=-1)(
                delayed(simulate)() for _ in range(n_runs)
            )

    filename = output_dir + "/" + filename_base + ".txt"
    output_data(filename, times)


def main() -> None:
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    run_experiment()


if __name__ == "__main__":
    main()
