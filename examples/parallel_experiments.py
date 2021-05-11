# Kyle Williams 3/5/20
from time_stepping import *
from boundary_geometry import RectangularDomain
from motion_model import BilliardMotion
from joblib import Parallel, delayed
import os

num_sensors: int = 20
sensing_radius: float = 0.2
timestep_size: float = 0.01

unit_square = RectangularDomain(spacing=sensing_radius)

billiard = BilliardMotion(domain=unit_square)

sensor_network = SensorNetwork(motion_model=billiard,
                               domain=unit_square,
                               sensing_radius=sensing_radius,
                               vel_mag=1,
                               n_sensors=num_sensors)

output_dir: str = "./output"
filename_base: str = "data"

n_runs: int = 5


# Unlike the animation, each simulation needs to create its own simulation object
def simulate() -> float:
    simulation = EvasionPathSimulation(sensor_network=sensor_network,
                                       dt=timestep_size)

    return simulation.run()


def output_data(filename: str, data_points: list) -> None:
    with open(filename, 'a+') as file:
        file.writelines("%.2f\n" % d for d in data_points)


def run_experiment() -> None:
    times = Parallel(n_jobs=-1)(
        delayed(simulate)() for _ in range(n_runs)
    )
    output_data(output_dir + "/" + filename_base + ".txt", times)


def main() -> None:
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    run_experiment()


if __name__ == "__main__":
    main()
