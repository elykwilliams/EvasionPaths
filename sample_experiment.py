from evasion_path import *
from statistics import *
from joblib import Parallel, delayed

# Setup
#   Step 1: Define problem parameters in run_experiment()
#   Step 2: Create Boundary and Motion Objects
#   Step 3: Create Evasion Paths Object, this is the main simulation object
#   Step 4: Set number of runs per data save, n_runs
#   Step 5: Set number of saves in main(), m_runs, for a total of m_runs*n_runs data points
#   Step 6: Set output directory, and data filename_base in main(),
#           ** Old data file with same name will be APPENDED to **
#           output directory is relative to this file location

# Run
#   run script as main

# Output
# There will be one output file in the selected output directory named filename_base.txt
# Each line will contain an element from the returned data points from run_experiment().
# Errors will be indicated, a seperate error-x.log file will appear if there were simulation
# errors with full details.
# Unhandled errors will be dumped to standard out; user can redirect if wished.


def run_experiment():
    n_sensors = 15
    sensing_radius = 0.2
    dt = 0.01

    unit_square = Boundary(spacing=sensing_radius)

    brownian_motion = BrownianMotion(dt=dt,
                                     sigma=0.01,
                                     sensing_radius=sensing_radius,
                                     boundary=unit_square)

    my_sim = EvasionPathSimulation(unit_square, brownian_motion, n_sensors, sensing_radius, dt)
    error_log = lambda i: "err-" + str(i) + ".log"

    n_runs = 5
    times = Parallel(n_jobs=-1)(delayed(simulate)(my_sim, error_log(i))
                                for i in range(n_runs))

    return times


def simulate(simulation, error_log=""):

    try:
        simulation.run()
    except GraphNotConnected:
        return "Graph not connected"
    except KeyError as err:
        with open(error_log, "a+") as file:
            file.write("Key Error Found\n")
            file.write("\tAttempted to find boundary cycle:"+str(err))
            file.write("\tDuring attempted:" + simulation.evasion_paths)
        return "Key Error"
    except MaxRecursionDepth as err:
        return "Max recursion depth exceeded"
    except InvalidStateChange as err:
        with open(error_log, "a+") as file:
            file.write("Unhandled state change error")
            file.write(str(err))
        return "Unhandled State Change"
    except AssertionError as err:
        with open(error_log, "a+") as file:
            file.write("Key Error Found")
            file.write("\tAttempted to find boundary cycle:" + str(err))
            file.write("\tDuring attempted:" + simulation.evasion_paths)
        return "Assert Error"
    except Exception as err:
        with open(error_log, "a+") as file:
            file.write("Unknown Error Occured" + str(err))
        return "Unknown Error"
    else:
        return simulation.time


def output_data(filename, data_points):
    with open(filename, 'a+') as file:
        for d in data_points:
            file.write(str(d) + "\n")


def main():
    output_dir = "./output"
    filename_base = "data"
    data = []
    m_runs = 3
    for run in range(m_runs):
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
