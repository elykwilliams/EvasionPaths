# Kyle Williams 3/4/20
import sys, os

sys.path.insert(1, '../src')

from evasion_path import *
from joblib import Parallel, delayed

# Setup
#   Set parameters at top of script
#   Motion Model or Boundary type are set in simulate()

# Run
#   python3 sample_experiment.py

# Output
#   There will be one output file in the selected output directory named filename_base.txt
#   Each line will contain an element from the returned data points from run_experiment() or
#       indicate an error.
#   Detailed error messages will be dumped to error-x.log where x is the jobid.
#   Unhandled errors will be dumped to standard out; user can redirect if wished.

num_sensors = 15
sensing_radius = 0.2
timestep_size = 0.01

output_dir = "./output"
filename_base = "data4"

n_runs = 100


def simulate():

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

    simulation.run()
    data = simulation.time

    return data


def output_data(filename, data_points):
    with open(filename, 'a+') as file:
        for d in data_points:
            file.write(str(d) + "\n")


def run_experiment():
    # This function calls simulate n_runs times in parallel
    times = Parallel(n_jobs=-1)(
        delayed(simulate)() for _ in range(n_runs)
    )

    filename = output_dir + "/" + filename_base + ".txt"
    output_data(filename, times)


def main():
    # Make sure output directory exists
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Run
    run_experiment()


if __name__ == "__main__":
    main()
