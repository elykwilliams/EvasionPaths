import matplotlib.pyplot as plt

Point = tuple[float]
Node = int
Edge = tuple[int]
Face = tuple[int]
Volume = tuple[int]


def plot_sensors(num, sensor_network):
    fig = plt.gcf()
    ax = plt.gca()

    ax.cla()

    # variable txt created to display the frame of the animation
    txt = fig.suptitle('')
    txt.set_text(f'num={num}')  # to keep track of the frames

    # clear axes and fix axis min,max values
    ax.set_xlim3d(-0.25, 1.25)
    ax.set_ylim3d(-0.25, 1.25)
    ax.set_zlim3d(-0.25, 1.25)

    # plot sensors
    ax.scatter(*sensor_network.mobile_sensors.T)
    ax.scatter(*sensor_network.fence_sensors.T)

    # draw arrows indicating magnitude and direction of movement
    ax.quiver(*sensor_network.mobile_sensors.T, *sensor_network.velocities.T, length=0.05)


def update_anim(num, simulation):
    simulation.do_timestep()
    simulation.time += simulation.dt
    plot_sensors(num, simulation.sensor_network)
    print(simulation.state)


def plot_edge(e: Edge, sensor_network):
    fig = plt.gcf()
    ax = plt.gca()
