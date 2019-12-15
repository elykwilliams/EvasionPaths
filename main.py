import gudhi
import networkx as nx
from combinatorial_map import *
from brownian_motion import *
import csv

from matplotlib.animation import FuncAnimation


def is_hole(graph, points, simplices):
    # Alpha shape is all points at boundary (this can be difined in initial setup.
    alpha_shape = [i for (i, p) in enumerate(points) if at_boundary(p)]
    if set(alpha_shape).issubset(set(graph.nodes())): return False
    if set(graph.nodes()) in simplices: return False

    return True
    # A hole cannot contatin the boundary, or be a simplex
    # return not ( set(alpha_shape).issubset(set(graph.nodes())) or graph.order() == 3)


filename = 'initial_data.csv'

with open(filename) as file:
    mypoints = [tuple(map(float, line.strip("\n").split(","))) for line in file]


def update(i):
    global mypoints # USe same points defined outside of function.
    mypoints = update_position(mypoints) # update via brownina motian

    # get alpha_complex
    alpha_complex = gudhi.AlphaComplex(mypoints)

    # Note, you can you alpha_square as input??
    # max_alpha_square = 0.04 ??
    simplex_tree = alpha_complex.create_simplex_tree()

    # get edges with filtration <= 0.04
    edges = [ simplex[0] for simplex in simplex_tree.get_skeleton(1) if len(simplex[0]) == 2 and simplex[1] <= 0.04]

    simplices = [set(simplex[0]) for simplex in simplex_tree.get_skeleton(2) if len(simplex[0]) == 3]
    # Use Edges to create graph
    G = nx.Graph()
    G.add_nodes_from(range(len(mypoints))) #nodes numbered 0 though N points -1
    G.add_edges_from(edges)

    # Use Graph and points to create cmap.
    cmap = CMap(G, mypoints)

    # filter out all cycles that are not holes(see function at top)
    holes = list(filter(lambda cycle:is_hole(cycle, mypoints, simplices), boundary_cycle_nodes(cmap)))
    
    # plot graph, and replot each hole but in red (since holes will be a list of graphs)
    ax.clear()
    nx.draw(G, mypoints)
    for h in holes:
        nx.draw(h, mypoints, node_color='red', edge_color="red", ax=ax)


# Animate
ax = plt.gca()
fig= plt.figure(1)

# repeat process 250 times
ani = FuncAnimation(fig, update, interval=1, frames=1)
ani.save('animation_working.gif', writer='imagemagick')

plt.show()



