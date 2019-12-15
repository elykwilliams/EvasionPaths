
# Kyle Williams, 11/25/2019

# Note that networkx does support planar-graph
# https://networkx.github.io/documentation/stable/reference/algorithms/planarity.html
# However, it appeaers that construction is more mannual,
# and that rotation information is stored as nodes/edges are added to the graph. 

import networkx as nx
from matplotlib import pyplot as plt
from math import atan2

# Sort counter-clockwise w.r.t center (Thanks to Nick Closuit for idea)
def theta(a, center):
    OA = (a[0] - center[0], a[1] - center[1])
    return atan2(OA[1], OA[0])


class CMap :
    def __init__(self, Graph, points):
        self.sorted_edges = dict()
        self.sorted_neighbors = dict()
        self.sorted_darts = dict()

        self.edge2dart = dict()
        self.dart2edge = dict()

        for i, edge in enumerate(Graph.edges()):
            self.edge2dart[edge] = 2*i
            self.edge2dart[(edge[1], edge[0])] = 2*i + 1

            self.dart2edge[2*i] = edge
            self.dart2edge[2*i+1] = (edge[1], edge[0])
            

        for node in Graph.nodes():
            neighbors = list(Graph.neighbors(node))

            # zip neighbors with associated coordinates for sorting
            neighbor_zip = zip(neighbors, [points[n] for n in neighbors])
            
            anticlockwise, clockwise = False, True

            # Sort
            sorted_zip = sorted(neighbor_zip, key = lambda pair: theta(pair[1], points[node]), reverse = anticlockwise)

            # Extract sorted edges
            self.sorted_edges[node] = [(node, n) for (n, _ ) in sorted_zip]

            # Not needed
            self.sorted_neighbors[node] = [n for (n, _ ) in sorted_zip]

            # Not 100% sure why e1, e2 need to be flipped to make things work
            self.sorted_darts[node] = [self.edge2dart[(e2, e1)] for e1,e2 in self.sorted_edges[node]]

        self.G = Graph
        self.points = points

        self.darts = list(range(2*Graph.size()))
        

    def sigma(self, dart):
        # Get node
        neighbor, node = self.dart2edge[dart]  # flip order?

        # Get index of given dart
        index = self.sorted_darts[node].index(dart)

        # Number of neighbors
        size = len(list(self.G.neighbors(node)))

        # Get next dart, wrap-around if out of range
        return  self.sorted_darts[node][(index+1) % size]


    def alpha(self, dart):
        # swap edge nodes, for other complement dart
        e1, e2 = self.dart2edge[dart]
        return self.edge2dart[(e2, e1)] 

    def phi(self, dart):
        return self.sigma(self.alpha(dart))

    def plot(self):
        nx.draw_networkx_labels(self.G, self.points)
        nx.draw_networkx_edge_labels(self.G, self.points, self.edge2dart, label_pos = 0.15)
        nx.draw(self.G, self.points)
        plt.show()

    def boundary_cycles(self):
        """
            This function returns a list of the boundary cycles by iterating on phi(x).
            The boundary cycles are given in terms of darts
            Boundary Cycles are guarenteed to be disjoint
            A cycle will have no repeated darts.
        """
        output = []
        all_darts = self.darts.copy()
        
        while(all_darts != []):
            # set root
            cycle = [all_darts.pop()]

            # get next in cycle
            next_dart = self.phi(cycle[0])

            while(next_dart != cycle[0]):

                # remove dart, since dijoint
                all_darts.remove(next_dart)

                # add to cycle
                cycle.append(next_dart)

                # get next dart
                next_dart = self.phi(next_dart)

            # cycle finised when next_dart is root   
            output.append(cycle)

        return output


def boundary_cycle_graphs(cmap):
    bcycles = []
    for cycle in cmap.boundary_cycles():
        simplex_edges = [cmap.dart2edge[dart] for dart in cycle]
        simplex_nodes = list(set((node for (node, _) in simplex_edges)))

        G = nx.Graph()
        G.add_nodes_from(simplex_nodes)
        G.add_edges_from(simplex_edges)

        bcycles.append(G)

    return bcycles


if __name__=="__main__":
    G = nx.house_x_graph()
    G.remove_edge(0, 1)
    G.remove_edge(2, 4)
    G.remove_edge(1, 3)
    G.add_edge(1, 0)
    G.add_edge(4, 0)
    #G = nx.gnm_random_graph(10, 15)
    points = [(cos(theta), sin(theta)) for theta in [2*pi*n/G.order() for n in range(G.order())]]

    cmap = CMap(G, points)

    print(cmap.boundary_cycles())

    simplices = boundary_cycle_nodes(cmap)

    boundary = [1, 2, 3, 4, 0]

    print([set(boundary) == set(G.nodes()) for G in simplices])

    for i, simplex in enumerate(simplices, start = 1):
        plot_num = str(2)+str(len(simplices)//2)+str(i)
        plt.subplot(int(plot_num))
        nx.draw_networkx_nodes(G, points, node_color="b")
        nx.draw_networkx_labels(G, points)
        nx.draw(simplex, points,node_color="r")

    plt.show()

    
    
