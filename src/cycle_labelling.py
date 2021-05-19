# ************************************************************
# Copyright (c) 2021, Kyle Williams - All Rights Reserved.
# You may use, distribute and modify this code under the
# terms of the BSD-3 license. You should have received a copy
# of the BSD-3 license with this file.
# If not, visit: https://opensource.org/licenses/BSD-3-Clause
# ************************************************************


import treelib.exceptions
from treelib import Tree

from topological_state import *
from utilities import *


## The CycleLabelling class manages the time dependant labelling of boundary cycles.
# The labelling adopts the following convention:
#
#       TRUE: there is a possible intruder within the boundary cycle
#       FALSE: there cannot be an intruder in the boundary cycle
#
# All of the updating is handled internally, and all of the information needed can be
# extracted from a given TopologicalState or StateChange.
#
# The labelling is updated according to the following rules
#
#       1. If a single 1-simplex is added splitting a single connected boundary cycle in two then
#           each new boundary cycle matches the original label. Remove the old boundary cycles from
#           the labelling.
#       2. If a single 1-simplex is removed joining two connected boundary cycle into one then
#          the new boundary cycle with have an intruder if either of the original boundary cycles
#          had an intruder. Remove the old boundary cycle from the labelling.
#       3. If a connected 2-simplex is added, the associated boundary cycle is marked false.
#       4. If a 2-simplex is removed, do nothing (label remains false).
#       5. If a connected 1-,2-simplex pair are added, add the 1-simplex according to (1), then add
#           the 2-simplex according to (3).
#       6. If a connected 1-,2-simplex pair is removed, update labelling according to (2).
#       7. If a Delaunay flip occurs, then label the added boundary cycles as false (since this can
#          only occur with 2-simplices). Remove the old boundary cycles from the labelling.
#       8. If a 1-simplex is removed resulting in a boundary cycle becoming disconnected, the connected
#           enclosing boundary cycle will be True if any of the disconnected boundary cycles were true or
#           the old boundary cycle was True. Remove the old boundary cycle and all disconnect boundary cycles
#           from the labelling.
#       9. If a 1-simplex is added resulting in a boundary cycle becoming connected, all newly added
#           boundary cycles will match the previously enclosing boundary cycle, and all cycles
#           that are 2-simplices will be labelled false. Remove the old enclosing boundary cycle.
#
class CycleLabelling:
    ## Initialize the cycle labeling for a given state.
    # The labelling is set in the following way:
    #
    #       All boundary cycles that are connected and not 2-simplices are labelled TRUE
    #       All connected 2-simplices are labelled as FALSE
    #       Any disconnected cycle has no label.
    #
    # Using the current 'forgetful' model, any cycle the becomes disconnected will be removed from
    # the labelling, and added back when it becomes reconnected.
    def __init__(self, state: TopologicalState) -> None:
        self._cycle_label = {cycle: True for cycle in state.boundary_cycles()}
        self._add_2simplex([state.simplex2cycle(s) for s in state.simplices(2) if state.is_connected_simplex(s)])
        self._delete_all([cycle for cycle in self._cycle_label.keys() if not state.is_connected_cycle(cycle)])

    ## Allow cycle labelling to be printable.
    # Used mostly for debugging
    def __str__(self):
        return ''.join([f'{key}:{val}\n' for key, val in self._cycle_label.items()])

    ## Check if cycle has a label.
    def __contains__(self, item):
        return item in self._cycle_label

    ## Protected access to cycle labelling.
    # The cycle labelling should be read only, and all updates managed internally.
    def __getitem__(self, item) -> bool:
        try:
            value = self._cycle_label[item]
        except KeyError:
            raise CycleNotFound(item)
        return value

    ## Check if any boundary cycles have an intruder.
    def has_intruder(self) -> bool:
        return any(self._cycle_label.values())

    ## Remove given boundary cycles from labelling.
    # Note that cycle_list should be an iterable list and not a
    # single boundary cycle.
    def _delete_all(self, cycle_list):
        for cycle in cycle_list:
            del self._cycle_label[cycle]

    ## Add edge.
    # When an edge is added, one boundary cycle is split into two
    # the two new boundary cycles are added, and the old boundary
    # cycle is removed.
    # Note that both parameters should be lists of boundary cycles.
    def _add_1simplex(self, removed_cycles, added_cycles):
        for cycle in added_cycles:
            self._cycle_label[cycle] = self._cycle_label[removed_cycles[0]]
        self._delete_all(removed_cycles)

    ## Remove edge.
    # When an edge is removed, two boundary cycles join into one.
    # The two old boundary cycles are removed, and the new boundary
    # cycle is added.
    # Note that both parameters should be lists of boundary cycles.
    def _remove_1simplex(self, removed_cycles, added_cycles):
        self._cycle_label[added_cycles[0]] = any([self._cycle_label[s] for s in removed_cycles])
        self._delete_all(removed_cycles)

    ## Mark boundary cycle as a 2-simplex.
    # added_simplices should be a list of boundary cycles corresponding to
    # the new 2-simplices
    def _add_2simplex(self, added_simplices):
        for simplex in added_simplices:
            self._cycle_label[simplex] = False

    ## Add 2-simplex + edge.
    # Need to add edge first so that added simple is in labelling.
    # removed_cycles and added_cycles should be lists of boundary cycles
    # added_simplex is the boundary cycle of the added simplex.
    def _add_simplex_pair(self, removed_cycles, added_cycles, added_simplices):
        self._add_1simplex(removed_cycles, added_cycles)
        self._add_2simplex(added_simplices)

    ## Remove edge and 2-simplex.
    # This is the same logic as removing just an edge.
    # removed_cycles and added_cycles should be lists of boundary cycles
    def _remove_simplex_pair(self, removed_cycles, added_cycles):
        self._remove_1simplex(removed_cycles, added_cycles)

    ## Delauny filp.
    # add 2 new simplices, remove two old simplices.
    # removed_cycles and added_cycles should be lists of boundary cycles
    def _delaunay_flip(self, removed_cycles, added_cycles):
        self._add_2simplex(added_cycles)
        self._delete_all(removed_cycles)

    ## Disconnect graph component - power-down model.
    # This can only happen through the removal of an edge.
    # should be removing all cycles that have become disconnected
    # the enclosing cycle is the connected boundary cycle which
    # encloses the removed cycles.
    def _disconnect(self, removed_cycles, enclosing_cycle):
        self._remove_1simplex(removed_cycles, [enclosing_cycle])

    ## Reconnect graph component.
    # This can only happen through the addition of an edge. Then,
    # adjust all 2-simplices.
    def _reconnect(self, added_cycles, connected_simplices, enclosing_cycle):
        self._add_1simplex([enclosing_cycle], added_cycles)
        self._add_2simplex(connected_simplices)

    ## Ignore state changes that involve disconnected boundary cycles.
    # Using the forgetful model, we must be careful to not operate on
    # cycles that have been disconnected which would at best raise a keylookup
    # error, and at worst, give incorrect results.
    #
    # Updates are ignored if they involve updates to any cycles that were not
    # previously in the labelling (i.e. disconnected). The one exception being the
    # case of a reconnection, in which case at least one of the cycles must be
    # disconnected (the cycle to be reconnected).
    def ignore_state_change(self, state_change):
        # No Change, or two isolated sensors becoming connected/disconnected
        if (
                state_change.case == (0, 0, 0, 0, 0, 0)
                or state_change.case == (1, 0, 0, 0, 1, 0)
                or state_change.case == (0, 1, 0, 0, 0, 1)
        ):
            return True
        # one or both old-cycle(s) were already disconnected
        if (
                state_change.case == (1, 0, 0, 0, 2, 1)
                or state_change.case == (1, 0, 1, 0, 2, 1)
                or state_change.case == (0, 1, 0, 0, 2, 1)
                or state_change.case == (0, 1, 0, 0, 1, 1)
                or state_change.case == (0, 1, 0, 0, 1, 2)
                or state_change.case == (0, 1, 0, 1, 1, 2)
                or state_change.case == (1, 1, 2, 2, 2, 2)
        ):
            return any([cycle not in self for cycle in state_change.cycles_removed])

        # disconnected cycle become 2-simplex, can't use above logic since no
        # cycles were removed.
        elif state_change.case == (0, 0, 1, 0, 0, 0):
            simplex = state_change.simplices_added[0]
            return not state_change.new_state.is_connected_simplex(simplex)

        # two disconnected components connecting/disconnecting
        elif (
                state_change.case == (1, 0, 0, 0, 1, 2)
                or state_change.case == (1, 0, 0, 0, 1, 1)
        ):
            return all([cycle not in self for cycle in state_change.cycles_removed])
        return False

    ## Update according to rules give.
    # Get cycles associated with any added simplices, and determine the enclosing
    # boundary cycle in the case of a disconnect or reconnect.
    def update(self, state_change):
        if not state_change.is_atomic():
            raise InvalidStateChange(state_change)

        if self.ignore_state_change(state_change):
            return

        # Add 1-Simplex
        elif state_change.case == (1, 0, 0, 0, 2, 1):
            self._add_1simplex(state_change.cycles_removed, state_change.cycles_added)

        # Remove 1-Simplex
        elif state_change.case == (0, 1, 0, 0, 1, 2):
            self._remove_1simplex(state_change.cycles_removed, state_change.cycles_added)

        # Add 2-Simplex
        elif state_change.case == (0, 0, 1, 0, 0, 0):
            simplex = state_change.simplices_added[0]
            added_simplices = [state_change.new_state.simplex2cycle(simplex)]
            self._add_2simplex(added_simplices)

        # Remove 2-Simplex
        elif state_change.case == (0, 0, 0, 1, 0, 0):
            return

        # 1-Simplex 2-Simplex Pair Added
        elif state_change.case == (1, 0, 1, 0, 2, 1):
            simplex = state_change.simplices_added[0]
            added_simplices = [state_change.new_state.simplex2cycle(simplex)]
            self._add_simplex_pair(state_change.cycles_removed, state_change.cycles_added, added_simplices)

        # 1-Simplex 2-Simplex Pair Removed
        elif state_change.case == (0, 1, 0, 1, 1, 2):
            self._remove_1simplex(state_change.cycles_removed, state_change.cycles_added)

        # Delaunay Flip
        elif state_change.case == (1, 1, 2, 2, 2, 2):
            self._delaunay_flip(state_change.cycles_removed, state_change.cycles_added)

        # cycle becoming disconnected resulting in inside and outside cycles
        # also isolated sensor becoming disconnected so only outer cycle
        elif state_change.case == (0, 1, 0, 0, 2, 1) or state_change.case == (0, 1, 0, 0, 1, 1):
            # Outer cycle will be the one that is connected
            enclosing_cycle = state_change.cycles_added[0]
            if (not state_change.new_state.is_connected_cycle(enclosing_cycle)
                    and len(state_change.cycles_added) != 1):
                enclosing_cycle = state_change.cycles_added[1]

            # cycles that have labelling but are not connected need to be removed
            disconnected_cycles = []
            for cycle in state_change.new_state.boundary_cycles():
                if not state_change.new_state.is_connected_cycle(cycle) and cycle in self:
                    disconnected_cycles.append(cycle)

            self._disconnect(state_change.cycles_removed + disconnected_cycles, enclosing_cycle)

        # cycle becoming reconnected resulting form inside and outside cycles joining by edge
        # also isolated sensor becoming connected so no inner cycle
        elif state_change.case == (1, 0, 0, 0, 1, 2) or state_change.case == (1, 0, 0, 0, 1, 1):
            # the outer cycle is the one which is connected
            enclosing_cycle = state_change.cycles_removed[0]
            if enclosing_cycle not in self and len(state_change.cycles_removed) != 1:
                enclosing_cycle = state_change.cycles_removed[1]

            # new cycles with no label should be added
            reconnected_cycles = []
            for cycle in state_change.new_state.boundary_cycles():
                if state_change.new_state.is_connected_cycle(cycle) and cycle not in self:
                    reconnected_cycles.append(cycle)

            # get boundary cycles for all connected simplices to be marked as clear
            # not just new ones
            connected_simplices = []
            for simplex in state_change.new_state.simplices(2):
                if state_change.new_state.is_connected_simplex(simplex):
                    simplex_cycle = state_change.new_state.simplex2cycle(simplex)
                    connected_simplices.append(simplex_cycle)

            self._reconnect(state_change.cycles_added + reconnected_cycles,
                            connected_simplices,
                            enclosing_cycle)


class CycleLabellingTree:

    ## Initialize all boundary cycles as True, simplices as False.
    # TODO Allow initially disconnected boundary cycles, or skip disconnected cycles
    # This can be adjusted by looking at the connected components in
    # topology and extracting the connected boundary cycles.
    def __init__(self, topology, method="power-down") -> None:
        self._tree = Tree()

        self.add_new_cycle(topology.alpha_cycle, parent=None)
        self.set(topology.alpha_cycle, False)

        for cycle in topology.boundary_cycles():
            self.add_new_cycle(cycle, topology.alpha_cycle)

        self.add_2simplices(topology.simplices(2))

        if method == "power-down" or method == "connected":
            self.remove_all(cycle for cycle in topology.boundary_cycles() if not topology.is_connected_cycle(cycle))

    def __contains__(self, item):
        return self._tree.contains(item)

    def __iter__(self):
        return self._tree.expand_tree(self._tree.root)

    def __getitem__(self, item):
        try:
            return self._tree[item].data
        except treelib.exceptions.NodeIDAbsentError as err:
            raise KeyError(f"Boundary Cycle {item} not found. You are attempting to retrieve the value of a cycle that "
                           f"has not yet been added to the tree.")
        except Exception as e:
            print(type(e))

    ## Add cycle to tree
    # defaults to True
    def add_new_cycle(self, cycle, parent):
        self._tree.create_node(tag=cycle, identifier=cycle, parent=parent, data=True)

    ## Set label.
    def set(self, cycle, value):
        try:
            self._tree.update_node(cycle, **{'data': value})
        except treelib.exceptions.NodeIDAbsentError as err:
            raise KeyError(f"Boundary Cycle {cycle} not found. "
                           f"You are attempting to change the value of a cycle that has not yet been added to the tree."
                           )

    ## Remove cycles from tree.
    def remove_all(self, cycles):
        for cycle in cycles:
            self._tree.remove_node(cycle)

    ## Set simplex cycle to be false
    # assumes cycles is already in tree
    def add_2simplices(self, added_cycles):
        for cycle in added_cycles:
            self.set(cycle, False)

    ## simplex that is no longer simplex, must sill be clear
    # code does nothing, but flags error if trying to remove non-cycle
    def remove_2simplices(self, removed_cycles):
        for cycle in removed_cycles:
            if cycle not in self:
                raise KeyError(f"Boundary Cycle {cycle} not found."
                               f"You are attempting to remove a simplex that has not yet been added to the tree")

    ## Add edge.
    # removed cycles list should have only 1 or 2 cycles
    # added_cycles should be a list with only 1 cycle
    # added cycle will have intruder if either removed cycle has intruder
    # This should only ever result in one cycle being removed
    def add_1simplex(self, removed_cycles, added_cycles):
        assert len(removed_cycles) == 1, "Adding edge cannot cause more than 1 boundary cycle to be removed"
        for cycle in added_cycles:
            self.add_new_cycle(cycle, self._tree.root)
            self.set(cycle, self[removed_cycles[0]])

        self.remove_all(removed_cycles)

    ## Remove edge.
    # This should only ever result in 1 cycle being added
    def remove_1simplex(self, removed_cycles, added_cycles):
        assert len(added_cycles) == 1, "Adding edge cannot cause more than 1 boundary cycle to be removed"
        cycle = added_cycles[0]
        self.add_new_cycle(cycle, self._tree.root)
        self.set(cycle, any([self[cycle] for cycle in removed_cycles]))

        self.remove_all(removed_cycles)
