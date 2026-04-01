# ******************************************************************************
#  Copyright (c) 2026, Contributors - All Rights Reserved.
# ******************************************************************************
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from boundary_geometry import SquareAnnulusDomain
from motion_model import BilliardMotion
from plotting_tools import show_domain_boundary, show_state
from sensor_network import SensorNetwork, generate_annulus_fence_sensors, generate_mobile_sensors
from time_stepping import EvasionPathSimulation


num_sensors = 36
sensing_radius = 0.4
timestep_size = 0.01
sensor_velocity = 1.0

filename_base = "SampleAnimationAnnulus"

domain = SquareAnnulusDomain(sensing_radius)
motion_model = BilliardMotion()

fence_bundle = generate_annulus_fence_sensors(domain, sensing_radius)
mobile_sensors = generate_mobile_sensors(domain, num_sensors, sensing_radius, sensor_velocity)

sensor_network = SensorNetwork(
    mobile_sensors,
    motion_model,
    fence_bundle.sensors,
    sensing_radius,
    domain,
    fence_groups=fence_bundle.fence_groups,
    excluded_fence_groups=fence_bundle.excluded_fence_groups,
)

simulation = EvasionPathSimulation(sensor_network, timestep_size)


def update(_):
    if not simulation.cycle_label.has_intruder():
        return

    simulation.do_timestep()

    axis = plt.gca()
    axis.cla()
    axis.axis("off")
    axis.set_aspect("equal", adjustable="box")
    pad = 2.0 * sensing_radius
    axis.set_xlim(-domain.outer_half_side - pad, domain.outer_half_side + pad)
    axis.set_ylim(-domain.outer_half_side - pad, domain.outer_half_side + pad)
    axis.set_title(f"T = {simulation.time:5.2f}", loc="left")

    show_domain_boundary(domain, ax=axis, color="k", linewidth=1.5)
    show_state(simulation, ax=axis)


if __name__ == "__main__":
    n_steps = 300
    ms_per_frame = 5000 * timestep_size

    fig = plt.figure(1)
    ani = FuncAnimation(fig, update, interval=ms_per_frame, frames=n_steps)

    plt.show()
    # ani.save(filename_base + ".mp4")
