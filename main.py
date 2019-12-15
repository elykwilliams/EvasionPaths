import gudhi
import networkx as nx
from combinatorial_map import *
from brownian_motion import *
import time

from matplotlib.animation import FuncAnimation

filename = 'initial_data.csv'

with open(filename) as file:
    mypoints = [tuple(map(float, line.strip("\n").split(","))) for line in file]

alpha_shape = [i for (i, p) in enumerate(mypoints) if at_boundary(p)]


def is_hole(graph):
    # A hole cannot contain the boundary, or be a simplex
    return not (set(alpha_shape).issubset(set(graph.nodes())) or graph.order() == 3)


def update(i):
    global mypoints  # Use same points defined outside of function.
    mypoints = update_position(mypoints)  # update via brownian motion

    # get alpha_complex
    alpha_complex = gudhi.AlphaComplex(mypoints)

    sensing_radius = 0.015
    max_alpha_square = 2*sensing_radius
    simplex_tree = alpha_complex.create_simplex_tree(max_alpha_square)

    # Get 1-simplices as edges for graph
    edges = [simplex for (simplex, _) in simplex_tree.get_skeleton(1) if len(simplex) == 2]

    # Use Edges to create graph
    G = nx.Graph()
    G.add_nodes_from(range(len(mypoints)))  # nodes numbered 0 though N points -1
    G.add_edges_from(edges)

    # Use Graph and points to create cmap.
    cmap = CMap(G, mypoints)

    # filter out all cycles that are not holes(see function at top)
    holes = list(filter(lambda cycle: is_hole(cycle), boundary_cycle_nodes(cmap)))
    
    # plot graph, and replot each hole but in red (since holes will be a list of graphs)
    ax.clear()
    nx.draw(G, mypoints)
    for h in holes:
       nx.draw(h, mypoints, node_color='red', edge_color="red", ax=ax)


# Animate
ax = plt.gca()
fig= plt.figure(1)

# repeat process 250 times
ani = FuncAnimation(fig, update, interval=100, frames=1)
ani.save('animation_working.gif', writer='imagemagick')

plt.show()



