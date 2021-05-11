# ************************************************************
# Copyright (c) 2021, Kyle Williams - All Rights Reserved.
# You may use, distribute and modify this code under the
# terms of the BSD-3 license. You should have received a copy
# of the BSD-3 license with this file.
# If not, visit: https://opensource.org/licenses/BSD-3-Clause
# ************************************************************


from time_stepping import *
from utilities import *
from boundary_geometry import RectangularDomain
from motion_model import BilliardMotion

from joblib import Parallel, delayed
import signal
import os

## In cases where it is unknown whether a simulation will terminate or not, you may
# want to set a timer on the simulation so it won't run longer that a set amount of time.
# This example shows how to do that on a linux machine. This script does not work on windows.

num_sensors: int = 20
sensing_radius: float = 0.2
timestep_size: float = 0.01

unit_square = RectangularDomain(spacing=sensing_radius)

billiard = BilliardMotion(domain=unit_square)

sensor_network = SensorNetwork(motion_model=billiard,
                               domain=unit_square,
                               sensing_radius=sensing_radius,
                               n_sensors=num_sensors,
                               vel_mag=1)

output_dir: str = "./output"
filename_base: str = "data"

n_runs: int = 1000
max_time: int = 10  # time in seconds


def handler(*_):
    raise TimedOutExc


def simulate() -> float:

    simulation = EvasionPathSimulation(sensor_network=sensor_network, dt=timestep_size)

    try:
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(max_time)
        simulation.run()
        data = simulation.time

    # Catch internal errors
    except EvasionPathError as e:
        data = str(e)

    # Catch all other errors
    except Exception as e:
        data = str(e)
    # Reset sigalarm
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
