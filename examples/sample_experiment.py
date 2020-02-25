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
filename_base = "data"

runs_per_save = 5
n_checkpoints = 1


def simulate(jobid):

    unit_square = Boundary(spacing=sensing_radius)

    brownian_motion = BrownianMotion(dt=dt,
                                     sigma=0.01,
                                     sensing_radius=sensing_radius,
                                     boundary=unit_square)

    simulation = EvasionPathSimulation(boundary=unit_square,
                                       motion_model=brownian_motion,
                                       n_sensors=n_sensors,
                                       sensing_radius=sensing_radius,
                                       dt=dt)

    error_log = output_dir+"/error-" + str(jobid) + ".log"

    try:
        simulation.run()
    except GraphNotConnected:
        return "Graph not connected"

    except KeyError as err:
        with open(error_log, "a+") as file:
            file.write("Key Error Found\n")
            file.write("\tAttempted to find boundary cycle:" + str(err)+"\n")
            file.write("\tDuring attempted:" + simulation.evasion_paths+"\n")
        return "Key Error"

    except MaxRecursionDepth as err:
        return "Max recursion depth exceeded"

    except InvalidStateChange as err:
        with open(error_log, "a+") as file:
            file.write("Unhandled state change error"+"\n")
            file.write(str(err)+"\n")
        return "Unhandled State Change"

    except AssertionError as err:
        with open(error_log, "a+") as file:
            file.write("Key Error Found"+"\n")
            file.write("\tAttempted to find boundary cycle:" + str(err)+"\n")
            file.write("\tDuring attempted:" + simulation.evasion_paths+"\n")
        return "Assert Error"

    except Exception as err:
        with open(error_log, "a+") as file:
            file.write("Unknown Error Occured" + str(err)+"\n")
        return "Unknown Error"

    else:
        return simulation.time


def run_experiment():
    times = \
        Parallel(n_jobs=-1)(
            delayed(simulate)(i)
            for i in range(runs_per_save)
        )
    return times


def output_data(filename, data_points):
    with open(filename, 'a+') as file:
        for d in data_points:
            file.write(str(d) + "\n")


def main():
    data = []
    for run in range(n_checkpoints):
        file_name = output_dir + "/" + filename_base + ".txt"
        try:
            data = run_experiment()
        except Exception as e:
            print("Experiment #" + str(run) + "has failed")
            print(type(e), e)
            print("Dumping Data:")
            print(data)
        else:
            try:
                output_data(file_name, data)
            except Exception as e:
                print("Output Failed")
                print(type(e), e)
                print("Dumping Data:")
                print(data)


if __name__ == "__main__":
    main()
