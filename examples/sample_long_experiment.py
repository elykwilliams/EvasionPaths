# Kyle Williams 3/5/20
import os, sys
# If you're having difficulty importing modules from the src directory, uncomment below
# current_dir = os.path.dirname(os.path.realpath(__file__))
# src_dir = os.path.join(current_dir, "..", "src")
# sys.path.append(src_dir)

from time_stepping import *
from joblib import Parallel, delayed

############################################################
## When running a simulation that will run for a long time, care must be taken to
# make sure that the simulation to not exit out in the middle, and that if there are
# errors, we know what they were. This example shows how to catch those errors



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
n_runs: int = 10
output_dir: str = "./output"
filename_base: str = "data"



############################################################
# Run the simulation
# - Unlike the animation, each simulation needs to create its own simulation object
# - The number of simulations that will be run at one time is defined by n_jobs. n_jobs=-1 will run as many as possible. 
def simulate() -> float:

    simulation = EvasionPathSimulation(boundary=my_boundary,
                                       motion_model=my_motion_model,
                                       n_int_sensors=num_sensors,
                                       sensing_radius=sensing_radius,
                                       dt=timestep_size)

    try:
        data = simulation.run()

    # Error from two changes happening simultaneously
    except MaxRecursionDepth as e:
        data = "Max recursion depth exceeded" + str(e.state_change.case)

    # Error from cycle not found in labelling (this should not happen)
    except KeyError as e:
        data = "Key Error" + str(e)

    # Found state change that was unhandled. Can happen when sensor escapes
    # or otherwise messes up boundary
    except InvalidStateChange as e:
        data = "Unhandled State Change" + str(e.state_change.case)

    # Catch all other errors
    except Exception as e:
        data = str(e)

    return data


def output_data(filename: str, data_points: list) -> None:
    with open(filename, 'a+') as file:
        for d in data_points:
            if type(d) != str:
                file.writelines("%.2f\n" % d)
            else:
                file.writelines(str(d) + "\n")


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



############################################################
if __name__ == "__main__":
    main()
