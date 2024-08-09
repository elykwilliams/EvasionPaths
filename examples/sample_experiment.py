# Kyle Williams 3/5/20

import os, sys
# If you're having difficulty importing modules from the src directory, uncomment below
# current_dir = os.path.dirname(os.path.realpath(__file__))
# src_dir = os.path.join(current_dir, "..", "src")
# sys.path.append(src_dir)

from evasionpaths.time_stepping import *


############################################################
# Define the parameters of the simulation
num_sensors: int = 20
sensing_radius: float = 0.2
timestep_size: float = 0.01

unit_square: Boundary = RectangularDomain(spacing=sensing_radius)

my_boundary = unit_square



############################################################
# Define the motion model -- see src/motion_model.py for more details
billiard: MotionModel = BilliardMotion(dt=timestep_size, 
                                       boundary=my_boundary, 
                                       vel=1, 
                                       n_int_sensors=num_sensors)

# See the paper for the Dorsogna model to better understand the parameters
dorsogna_coeff = {"Ca": 0.45, "la": 1, "Cr": 0.5, "lr": 0.1}
dorsogna: MotionModel = Dorsogna(dt=timestep_size, 
                                 boundary=my_boundary, 
                                 max_vel=1, 
                                 n_int_sensors=num_sensors, 
                                 sensing_radius=sensing_radius, 
                                 DO_coeff=dorsogna_coeff)

brownian: MotionModel = BrownianMotion(dt=timestep_size,
                                        boundary=my_boundary,
                                        sigma=0.5)

##############################
# ASSIGN MOTION MODEL HERE
my_motion_model = billiard



############################################################
# Define how many simulations to run and where to save the resulting times
# - The simulations will be run sequentially. See parallel_experiments.py for parallel runs
n_runs: int = 1
output_dir: str = "./output"
filename_base: str = "data"



############################################################
# Run the simulation
# - Unlike the animation, each simulation needs to create its own simulation object
def simulate() -> float:

    simulation = EvasionPathSimulation(boundary=my_boundary,
                                       motion_model=my_motion_model,
                                       n_int_sensors=num_sensors,
                                       sensing_radius=sensing_radius,
                                       dt=timestep_size)


    return simulation.run()


def output_data(filename: str, data_points: list) -> None:
    with open(filename, 'a+') as file:
        for d in data_points:
            if type(d) != str:
                file.writelines("%.2f\n" % d)
            else:
                file.writelines(str(d) + "\n")


def run_experiment() -> None:
    times = [simulate() for _ in range(n_runs)]
    filename = output_dir + "/" + filename_base + ".txt"
    output_data(filename, times)


def main() -> None:
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    run_experiment()



############################################################
if __name__ == "__main__":
    main()
