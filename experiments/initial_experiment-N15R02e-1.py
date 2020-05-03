# Kyle Williams 2/25/20
import sys, os
sys.path.insert(1, '../src')

from evasion_path import *
from statistics import *
from joblib import Parallel, delayed


# Setup
#   Set parameters at top of script
#   Motion Model or Boundary type are set in simulate()

# Run
#   python sample_experiment.py

# Output
#   There will be one output file in the selected output directory named filename_base.txt
#   Each line will contain an element from the returned data points from run_experiment() or
#       indicate an error.
#   Detailed error messages will be dumped to error-x.log where x is the jobid.
#   Unhandled errors will be dumped to standard out; user can redirect if wished.

n_sensors = 15
sensing_radius = 0.2
dt = 0.01

output_dir = "./output"
filename_base = "data4"

n_runs = 10000


def simulate(jobid):

    unit_square = RectangularDomain(spacing=sensing_radius)

    brownian_motion = BrownianMotion(dt=dt,
                                     sigma=0.01,
                                     sensing_radius=sensing_radius,
                                     boundary=unit_square)

    simulation = EvasionPathSimulation(boundary=unit_square,
                                       motion_model=brownian_motion,
                                       n_sensors=n_sensors,
                                       sensing_radius=sensing_radius,
                                       dt=dt)

    error_log = output_dir + "/error-" + str(jobid) + ".log"
    filename = output_dir + "/" + filename_base + "-" + str(jobid) + ".txt"
    
    try:
        simulation.run()
    except GraphNotConnected:
        data = "Graph not connected"

    except MaxRecursionDepth as err:
        data = "Max recursion depth exceeded"

    except KeyError as err:
        with open(error_log, "a+") as file:
            file.write("Key Error Found\n")
            file.write("\tAttempted to find boundary cycle:" + str(err)+"\n")
            file.write("\tDuring attempted:" + simulation.evasion_paths+"\n")
        data = "Key Error"
        
    except InvalidStateChange as err:
        with open(error_log, "a+") as file:
            file.write("Unhandled state change error"+"\n")
            file.write(str(err)+"\n")
        data = "Unhandled State Change"

    except AssertionError as err:
        with open(error_log, "a+") as file:
            file.write("Key Error Found"+"\n")
            file.write("\tAttempted to find boundary cycle:" + str(err)+"\n")
            file.write("\tDuring attempted:" + simulation.evasion_paths+"\n")
        data = "Assert Error"

    except Exception as err:
        with open(error_log, "a+") as file:
            file.write("Unknown Error Occured" + str(err)+"\n")
        data = "Unknown Error"

    else:
        data = simulation.time

    try:
        with open(filename, "a+") as file:
            file.write(str(data) + "\n")
    except IOError as e:
        pass

def run_experiment():
    times = \
        Parallel(n_jobs=-1)(
            delayed(simulate)(i)
            for i in range(n_runs)
        )
    return times


def output_data(filename, data_points):
    with open(filename, 'a+') as file:
        for d in data_points:
            file.write(str(d) + "\n")


def main():
    data = []
  
    # Make sure output directory exists
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    run_experiment()



if __name__ == "__main__":
    main()
