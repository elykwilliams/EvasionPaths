from brownian_motion import *
from numpy import sqrt
from gudhi import AlphaComplex


class EvasionPathSimulation:
    def __init__(self, dt, end_time):
        # Parameters
        self.n_interior_sensors = 30
        self.sensing_radius = 0.015
        self.dt = dt
        self.Tend = end_time
        self.time = 0
        self.brownian_motion = BrownianMotion(self.dt, sigma=0.1)

        # Point data
        self.points = self.brownian_motion.generate_points(self.n_interior_sensors, self.sensing_radius)
        self.n_sensors = len(self.points)
        self.alpha_shape = list(range(self.brownian_motion.boundary.n_points))

        # Complex info
        self.alpha_complex = None
        self.simplex_tree = None
        self.edges = None
        self.faces = None

        self.G = None
        self.cmap = None

        # Complex Coloring
        self.cell_coloring = None
        self.holes = None
        self.evasion_paths = None

    def run(self):
        if self.Tend > self.dt:
            while self.time < self.Tend:
                self.time += self.dt
                self.do_timestep()
                print(self.time)
            return bool(self.evasion_paths)
        else:
            while self.evasion_paths:
                self.time += self.dt
                self.do_timestep()
            return self.time




    def do_timestep(self):

        # Update Points
        self.points = self.brownian_motion.update_points(self.points)

        # Create Alpha Complex
        self.alpha_complex = AlphaComplex(self.points)
        self.simplex_tree = self.alpha_complex.create_simplex_tree(2 * self.sensing_radius)

        # Build Graph
        self.edges = [simplex for simplex, _ in self.simplex_tree.get_skeleton(1) if len(simplex) == 2]
        self.faces = [simplex for simplex, _ in self.simplex_tree.get_skeleton(2) if len(simplex) == 3]

        self.G = nx.Graph()
        self.G.add_nodes_from(range(self.n_sensors))  # nodes numbered 0 though N points -1
        self.G.add_edges_from(self.edges)

        # Build Combinatorial Map
        self.cmap = CMap(self.G, self.points)

        # Find Holes
        self.find_holes()

        # Find Evasion Path
        self.find_evasion_paths()

    def find_holes(self):
        boundary_cycles = boundary_cycle_graphs(self.cmap)
        self.holes = list(filter(lambda cycle: is_hole(cycle, self.alpha_shape), boundary_cycles))
        # self.cell_coloring = dict(zip(boundary_cycles, coloring))

    def find_evasion_paths(self):
        pass


if __name__ == "__main__":
    simplex = EvasionPathSimulation()
    simplex.do_timestep()

    ax = plt.gca()
    fig = plt.figure(1)
    ax.clear()
    nx.draw(simplex.G, simplex.points)
    for h in simplex.holes:
        nx.draw(h, simplex.points, node_color='red', edge_color="red", ax=ax)

    simplex.do_timestep()

    for h in simplex.holes:
        nx.draw(h, simplex.points, node_color='green', edge_color="green", ax=ax)

    plt.show()