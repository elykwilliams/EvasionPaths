# Kyle Williams 3/5/20
import os
from time_stepping import *
from joblib import Parallel, delayed

num_sensors: int = 20
sensing_radius: float = 0.2
timestep_size: float = 0.01

unit_square: Domain = RectangularDomain(spacing=sensing_radius)

# noinspection PyTypeChecker
billiard: MotionModel = BilliardMotion(dt=timestep_size, boundary=unit_square, vel=1, n_int_sensors=num_sensors)

output_dir: str = "./output"
filename_base: str = "data"

n_runs: int = 5


# Unlike the animation, each simulation needs to create its own simulation object
def simulate() -> float:

    simulation = EvasionPathSimulation(boundary=unit_square,
                                       motion_model=billiard,
                                       n_int_sensors=num_sensors,
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
    output_data(output_dir + "/" + filename_base + ".txt", times)


def main() -> None:
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    run_experiment()


if __name__ == "__main__":
    main()
