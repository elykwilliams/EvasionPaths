from sensor_network import SensorNetwork
from topological_state import TopologicalState, StateChange


class Simulation:
    def __init__(self, sensor_network: SensorNetwork, dt: float, end_time: float = 0) -> None:
        self.dt = dt
        self.Tend = end_time
        self.time = 0

        self.sensor_network = sensor_network
        self.state = TopologicalState(self.sensor_network)

    def run(self) -> None:
        while self.time < self.Tend:
            print(f"Time: {self.time}")
            self.do_timestep()
            self.time += self.dt

    def do_timestep(self, level: int = 0) -> None:
        print(f"Level: {level}")
        dt = self.dt * 2 ** -level

        for _ in range(2):
            self.sensor_network.move(dt)
            new_state = TopologicalState(self.sensor_network)
            state_change = StateChange(new_state, self.state)

            if state_change.is_atomic():
                self.update(new_state)
            elif level + 1 == 25:
                self.save(state_change)
                # insert new case
                state_change.valid_cases.append(state_change.case)
            else:
                self.do_timestep(level=level + 1)

            if level == 0:
                return

    def update(self, new_state):
        self.sensor_network.update()
        self.state = new_state

    # save old and new topologies along with positions/embeddings
    def save(self, topology_change):
        filename = f"data/new_case_{len(topology_change.valid_cases)}.txt"

        with open(filename, "w+") as file:
            file.write(str(topology_change))
            file.write("\n\n")
            file.write(str(self.sensor_network))
