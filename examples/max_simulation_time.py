# Kyle Williams 3/5/20
import os
from evasionpaths.time_stepping import *
from joblib import Parallel, delayed
import signal

## In cases where it is unknown whether a simulation will terminate or not, you may
# want to set a timer on the simulation so it won't run longer that a set amount of time.
# This example shows how to do that.

num_sensors: int = 20
sensing_radius: float = 0.2
timestep_size: float = 0.01

unit_square: Boundary = RectangularDomain(spacing=sensing_radius)

# noinspection PyTypeChecker
billiard: MotionModel = BilliardMotion(dt=timestep_size, boundary=unit_square, vel=1, n_int_sensors=num_sensors)

output_dir: str = "./output"
filename_base: str = "data"

n_runs: int = 1000
max_time: int = 600  # time in seconds

class TimedOutExc(Exception):
    pass


def handler(signum, frame):
    raise TimedOutExc


def simulate() -> float:

    simulation = EvasionPathSimulation(boundary=unit_square,
                                       motion_model=billiard,
                                       n_int_sensors=num_sensors,
                                       sensing_radius=sensing_radius,
                                       dt=timestep_size)

    try:
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(max_time)
        simulation.run()
        data = simulation.time

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
    finally:
        signal.alarm(0)
    return data


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
