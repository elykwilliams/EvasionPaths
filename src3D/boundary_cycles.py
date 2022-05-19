import numpy as np


class OrientedSimplex:
    def __init__(self, nodes):
        self.nodes = tuple(nodes)
        self.dim = len(self.nodes) - 1

    def alpha(self):
        if self.dim == 1:
            return OrientedSimplex([self.nodes[1], self.nodes[0]])
        elif self.dim == 2:
            return OrientedSimplex([self.nodes[0], self.nodes[2], self.nodes[1]])

    def is_edge(self, half_edge):
        if not all([n in self.nodes for n in half_edge.nodes]):
            return False
        i = self.nodes.index(half_edge.nodes[0])
        return (self.nodes[i % 3], self.nodes[(i + 1) % 3]) == half_edge.nodes

    @property
    def edges(self):
        result = []
        for i in range(len(self.nodes)):
            half_edge = (self.nodes[i % 3], self.nodes[(i + 1) % 3])
            result.append(OrientedSimplex(half_edge))
        return result

    def vertices(self, points):
        return [points[n] for n in self.nodes]

    def orient(self, half_edge):
        i = self.nodes.index(half_edge.nodes[0])
        return OrientedSimplex(self.nodes[i:] + self.nodes[:i])

    def __hash__(self):
        i = 0 if self.dim == 1 else self.nodes.index(min(self.nodes))
        return hash(repr(OrientedSimplex(self.nodes[i:] + self.nodes[:i])))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __repr__(self):
        return "(" + ",".join(map(str, self.nodes)) + ")"


class CMap:
    def __init__(self, points, edges, simplices):
        self.points = points
        self.edges = edges
        self.simplices = simplices

        self.oriented_faces = get_oriented(self.simplices)
        self.half_edges = get_oriented(self.edges)

        self.rotinfo = dict()
        for half_edge in self.half_edges:
            temp = self.incident_simplices(half_edge)
            self.rotinfo[half_edge] = sorted(temp, key=lambda simplex: self.theta(temp[0], simplex))

        self.hashed_cycles = dict()

    def incident_simplices(self, half_edge: OrientedSimplex):
        return [simplex.orient(half_edge) for simplex in self.oriented_faces if simplex.is_edge(half_edge)]

    def alpha(self, face):
        return face.alpha()

    def sigma(self, face, half_edge):
        if not face.is_edge(half_edge):
            return  # maybe should be error???
        incident_faces = self.rotinfo[half_edge]

        for i in range(len(incident_faces)):
            if incident_faces[i] == face:
                return incident_faces[(i + 1) % len(incident_faces)]

    def phi(self, simplex, half_edge):
        return self.alpha(self.sigma(simplex, half_edge))

    def flop(self, simplices):
        result = set(simplices)
        for simplex in simplices:
            result.update([self.phi(simplex, e) for e in simplex.edges])
        return result

    def boundary_cycle(self, simplex):
        if simplex in self.hashed_cycles:
            return self.hashed_cycles[simplex]
        cycle = {simplex}
        while self.flop(cycle) != cycle:
            cycle = self.flop(cycle)
        self.hashed_cycles[simplex] = frozenset(cycle)
        return self.hashed_cycles[simplex]

    def get_boundary_cycles(self):
        faces = set(self.oriented_faces)
        cycles = []
        while faces:
            f = faces.pop()
            bcycle = self.boundary_cycle(f)
            faces.difference_update(bcycle)
            cycles.append(bcycle)
        return cycles

    def theta(self, simplex1, simplex2):
        # compute angle with respect to two_simplices[0]
        vertices1 = simplex1.vertices(self.points)
        vertices2 = simplex2.vertices(self.points)
        vector1 = vertices1[2] - vertices1[0]
        vector2 = vertices2[2] - vertices2[0]
        normal = vertices1[1] - vertices1[0]

        projected_vector1 = vector1 - (np.dot(vector1, normal)) / (np.linalg.norm(normal) ** 2) * normal
        projected_vector2 = vector2 - (np.dot(vector2, normal)) / (np.linalg.norm(normal) ** 2) * normal

        dot_prod = np.dot(projected_vector1, projected_vector2) / (
                np.linalg.norm(projected_vector1) * np.linalg.norm(projected_vector2))
        if np.abs(dot_prod) - 1 > 1e-8:
            print(f"Warning, truncating dot product from {dot_prod}")
        dot_prod = np.clip(dot_prod, -1.0, 1.0)

        if np.dot(normal, (np.cross(projected_vector1, projected_vector2))) >= 0:
            return -np.arccos(dot_prod)
        else:
            return np.arccos(dot_prod)


def get_oriented(simplices):
    oriented_simplices = []
    for simplex in simplices:
        os = OrientedSimplex(simplex)
        oriented_simplices.extend([os, os.alpha()])
    return oriented_simplices


def share_edge(simplex1, simplex2):
    return len(set(simplex1).intersection(set(simplex2))) == 2
