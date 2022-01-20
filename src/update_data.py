from topological_state import StateChange, TopologicalState
from utilities import *


## Data to preform Labelling Update.
# This class contains all the information needed to update the labelling tree.
# The structure is different from state change in that each subclass contains a method
# allowing it to compute the new cycle labels. This class is different from the statechange
# because the statechange only looks at the differences in topology, whereas updatadata also
# incorporates previous cycle labeling info. The label updates, are not static.
class UpdateData(StateChange):

    ## Initialize with cycle labelling and StateChange.
    # Make sure that state change is atomic, raises UpdateError Otherwise.
    def __init__(self, tree_labels, old_state: TopologicalState, new_state: TopologicalState):
        super().__init__(old_state, new_state)
        self.labels = tree_labels
        self.new_state = new_state

        try:
            self.is_valid()
        except UpdateError:
            raise

    ## Check if update is self-consistant.
    # i.e. it is the update it says it is.
    def is_valid(self):
        if not all(cycle in self.labels or cycle in self.cycles_added for cycle in self.simplices_added):
            raise UpdateError(f"You are attempting to add a simplex that has not yet been added to the tree")
        elif not all(cycle in self.labels for cycle in self.simplices_removed + self.cycles_removed):
            raise UpdateError(f"You are attempting to remove a simplex/cycle that has not yet been added to the tree")
        return self.is_atomic()

    # Return Mapping with labelling for each cycle with a new labelling.
    @property
    def label_update(self):
        return dict()


class Add2Simplices(UpdateData):
    ## All 2-Simplices are Labeled False
    @property
    def label_update(self):
        return {cycle: False for cycle in self.simplices_added}


class Remove2Simplices(UpdateData):
    #  No update needed, use default
    pass


class Add1Simplex(UpdateData):
    ## Add edge.
    # New cycles will match the removed cycle
    @property
    def label_update(self):
        return {cycle: self.labels[self.cycles_removed[0]] for cycle in self.cycles_added}


class Remove1Simplex(UpdateData):
    ## Remove edge.
    # added cycle will have intruder if either removed cycle has intruder
    @property
    def label_update(self):
        return {cycle: any([self.labels[cycle] for cycle in self.cycles_removed]) for cycle in self.cycles_added}



class AddSimplexPair(UpdateData):
    ## Add simplex pair
    # Will split a cycle into two with label as in Add1Simplex, then label 2simplex as False
    @property
    def label_update(self):
        return {cycle: False if cycle in self.simplices_added else self.labels[self.cycles_removed[0]]
                for cycle in self.cycles_added}

    # If a 1-simplex and 2-simplex are added/removed simultaniously, then the 1-simplex
    # should be an edge of the 2-simplex
    def is_atomic(self):
        # Check consistency with definition
        simplex = self.simplices_added[0]
        edge = self.edges_added[0]
        if not is_subset(edge, simplex):
            return False
        return True


class RemoveSimplexPair(Remove1Simplex):
    ## Same as Remove1Simplex
    def is_atomic(self):
        # If a 1-simplex and 2-simplex are added/removed simultaniously, then the 1-simplex
        # should be an edge of the 2-simplex
        simplex = self.simplices_removed[0]
        edge = self.edges_removed[0]
        if not is_subset(edge, simplex):
            return False
        return True



class DelaunyFlip(UpdateData):
    ## Delauny Flip.
    # edge between two simplices flips resulting in two new simplices
    # Note that this can only happen with simplices.
    @property
    def label_update(self):
        return {cycle: False for cycle in self.simplices_added}

    # The set of vertices of the 1-simplices should contain the vertices of each 2-simplex
    # that is added or removed.
    def is_atomic(self):
        old_edge = self.edges_removed[0]
        new_edge = self.edges_added[0]
        if not all([is_subset(old_edge, s) for s in self.simplices_removed]):
            return False
        elif not all([is_subset(new_edge, s) for s in self.simplices_added]):
            return False

        nodes = list(set(old_edge).union(set(new_edge)))
        if not all([is_subset(s, nodes) for s in self.simplices_removed]):
            return False
        elif not all([is_subset(s, nodes) for s in self.simplices_added]):
            return False
        return True


def get_label_update(old_labelling, old_state, new_state):
    # label_update will be defined to be a valid dictionary or of type InvalidStateChange
    case = StateChange(old_state, new_state).case
    temp = {
        (0, 0, 0, 0, 0, 0): UpdateData,  # No Change
        (0, 1, 0, 0, 1, 2): Remove1Simplex,
        (1, 0, 0, 0, 2, 1): Add1Simplex,
        (0, 0, 0, 1, 0, 0): Remove2Simplices,
        (0, 0, 1, 0, 0, 0): Add2Simplices,
        (1, 0, 1, 0, 2, 1): AddSimplexPair,
        (0, 1, 0, 1, 1, 2): RemoveSimplexPair,
        (1, 1, 2, 2, 2, 2): DelaunyFlip
    }
    try:
        return temp[state_change.case](labelling, state_change)
    except KeyError:
        raise UpdateError("The requested change is non-atomic, or results in disconnection")
    except UpdateError:
        raise
