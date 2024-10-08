# ************************************************************
# Copyright (c) 2020, Kyle Williams - All Rights Reserved.
# You may use, distribute and modify this code under the
# terms of the BSD-3 license. You should have received a copy
# of the BSD-3 license with this file.
# If not, visit: https://opensource.org/licenses/BSD-3-Clause
# ************************************************************


from evasionpaths.topological_state import *


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
    # Using the current "forgetful" model, any cycle the becomes disconnected will be removed from
    # the labelling, and added back when it becomes reconnected.
    def __init__(self, state: TopologicalState) -> None:
        self._cycle_label = dict()

        for cycle in state.boundary_cycles():
            self._cycle_label[cycle] = True
        for cycle in [state.simplex2cycle(s) for s in state.simplices(2) if state.is_connected_simplex(s)]:
            self._add_2simplex(cycle)
        self._delete_all([cycle for cycle in self._cycle_label.keys() if not state.is_connected_cycle(cycle)])

    ## Allow cycle labelling to be printable.
    # Used mostly for debugging
    def __str__(self):
        res = ""
        for key, val in self._cycle_label.items():
            res += str(key) + ": " + str(val) + "\n"
        return res

    ## Check if cycle has a label.
    def __contains__(self, item):
        return item in self._cycle_label

    ## Protected access to cycle labelling.
    # The cycle labelling should be read only, and all updates managed internally.
    def __getitem__(self, item):
        try:
            value = self._cycle_label[item]
        except KeyError:
            raise CycleNotFound(item)
        return value

    ## Check if any boundary cycles have an intruder.
    def has_intruder(self):
        return any(self._cycle_label.values())

    def _delete_all(self, cycle_list):
        for cycle in cycle_list:
            del self._cycle_label[cycle]

    def _add_1simplex(self, removed_cycles, added_cycles):
        for cycle in added_cycles:
            self._cycle_label[cycle] = self._cycle_label[removed_cycles[0]]
        self._delete_all(removed_cycles)

    def _remove_1simplex(self, removed_cycles, added_cycles):
        assert(len(added_cycles) == 1)

        self._cycle_label[added_cycles[0]] = any([self._cycle_label[s] for s in removed_cycles])
        self._delete_all(removed_cycles)

    def _add_2simplex(self, added_simplex):
        self._cycle_label[added_simplex] = False

    def _add_simplex_pair(self, removed_cycles, added_cycles, added_simplex):
        self._add_1simplex(removed_cycles, added_cycles)
        self._add_2simplex(added_simplex)

    def _remove_simplex_pair(self, removed_cycles, added_cycles):
        self._remove_1simplex(removed_cycles, added_cycles)

    def _delaunay_flip(self, removed_cycles, added_cycles):
        for cycle in added_cycles:
            self._add_2simplex(cycle)
        self._delete_all(removed_cycles)

    def _disconnect(self, removed_cycles, enclosing_cycle):
        self._remove_1simplex(removed_cycles, [enclosing_cycle])

    def _reconnect(self, added_cycles, enclosing_cycle, connected_simplices):
        self._add_1simplex([enclosing_cycle], added_cycles)
        for cycle in connected_simplices:
            self._add_2simplex(cycle)

    ## Ignore state changes that involve disconnected boundary cycles.
    # Using the forgetful model, we must be careful to not operate on
    # cycles that have been disconnected which would at best raise a keylookup
    # error, and at worst, give incorrect results.
    #
    # Updates are ignored if they involve updates to any cycles that were not
    # previously in the labelling or disconnected. The one exception being the
    # case of a reconnection, in which case at least one of the cycles must be
    # disconnected (the cycle to be reconnected).
    def ignore_state_change(self, state_change):
        # No Change
        if state_change.case == (0, 0, 0, 0, 0, 0) \
                or state_change.case == (1, 0, 0, 0, 1, 0) \
                or state_change.case == (0, 1, 0, 0, 0, 1):
            return True
        # one or both old-cycle is disconnected
        if state_change.case == (1, 0, 0, 0, 2, 1) \
                or state_change.case == (1, 0, 1, 0, 2, 1) \
                or state_change.case == (0, 1, 0, 0, 2, 1) \
                or state_change.case == (0, 1, 0, 0, 1, 1) \
                or state_change.case == (0, 1, 0, 0, 1, 2) \
                or state_change.case == (0, 1, 0, 1, 1, 2) \
                or state_change.case == (1, 1, 2, 2, 2, 2):
            return any([cell not in self._cycle_label for cell in state_change.cycles_removed])
        # simplex-cycle is disconnected
        elif state_change.case == (0, 0, 1, 0, 0, 0):
            simplex = state_change.simplices_added[0]
            return not state_change.new_state.is_connected_simplex(simplex)
        # enclosing-cycle is disconnected
        elif state_change.case == (1, 0, 0, 0, 1, 2) \
                or state_change.case == (1, 0, 0, 0, 1, 1):
            return all([cycle not in self._cycle_label for cycle in state_change.cycles_removed])
        return False

    ## Update according to rules give.
    # Get cycles associated with any added simplices, and determine the enclosing
    # boundary cycle in the case of a disconnect or reconnect.
    def update(self, state_change):
        if not state_change.is_atomic():
            raise InvalidStateChange(state_change)

        if self.ignore_state_change(state_change):
            return ""

        # Add 1-Simplex
        elif state_change.case == (1, 0, 0, 0, 2, 1):
            self._add_1simplex(state_change.cycles_removed, state_change.cycles_added)

        # Remove 1-Simplex
        elif state_change.case == (0, 1, 0, 0, 1, 2):
            self._remove_1simplex(state_change.cycles_removed, state_change.cycles_added)

        # Add 2-Simplex
        elif state_change.case == (0, 0, 1, 0, 0, 0):
            simplex = state_change.simplices_added[0]
            added_simplex = state_change.new_state.simplex2cycle(simplex)
            self._add_2simplex(added_simplex)

        # Remove 2-Simplex
        elif state_change.case == (0, 0, 0, 1, 0, 0):
            return ""

        # 1-Simplex 2-Simplex Pair Added
        elif state_change.case == (1, 0, 1, 0, 2, 1):
            simplex = state_change.simplices_added[0]
            added_simplex = state_change.new_state.simplex2cycle(simplex)
            self._add_simplex_pair(state_change.cycles_removed, state_change.cycles_added, added_simplex)

        # 1-Simplex 2-Simplex Pair Removed
        elif state_change.case == (0, 1, 0, 1, 1, 2):
            self._remove_1simplex(state_change.cycles_removed, state_change.cycles_added)

        # Delaunay Flip
        elif state_change.case == (1, 1, 2, 2, 2, 2):
            self._delaunay_flip(state_change.cycles_removed, state_change.cycles_added)

        # Disconnect
        elif state_change.case == (0, 1, 0, 0, 2, 1) or state_change.case == (0, 1, 0, 0, 1, 1):
            enclosing_cycle = state_change.cycles_added[0]
            if not state_change.new_state.is_connected_cycle(enclosing_cycle) \
                    and len(state_change.cycles_added) != 1:
                enclosing_cycle = state_change.cycles_added[1]

            disconnected_cycles = []
            for cycle in state_change.new_state.boundary_cycles():
                if not state_change.new_state.is_connected_cycle(cycle) and cycle in self._cycle_label:
                    disconnected_cycles.append(cycle)

            self._disconnect(state_change.cycles_removed + disconnected_cycles, enclosing_cycle)

        # Reconnect
        elif state_change.case == (1, 0, 0, 0, 1, 2) or state_change.case == (1, 0, 0, 0, 1, 1):
            enclosing_cycle = state_change.cycles_removed[0]
            if enclosing_cycle not in self._cycle_label and len(state_change.cycles_removed) != 1:
                enclosing_cycle = state_change.cycles_removed[1]

            reconnected_cycles = []
            for cycle in state_change.new_state.boundary_cycles():
                if state_change.new_state.is_connected_cycle(cycle) and cycle not in self._cycle_label:
                    reconnected_cycles.append(cycle)

            connected_simplices = []
            for simplex in state_change.new_state.simplices(2):
                if state_change.new_state.is_connected_simplex(simplex):
                    simplex_cycle = state_change.new_state.simplex2cycle(simplex)
                    connected_simplices.append(simplex_cycle)

            self._reconnect(state_change.cycles_added + reconnected_cycles, enclosing_cycle,
                            connected_simplices)

        return StateChange.case2name[state_change.case]


class CycleNotFound:
    def __init__(self, boundary_cycle):
        self.b = boundary_cycle

    def __str__(self):
        return "Attempted to retrieve labelling for " + str(self.b) + ", " \
                 "but this cycle was not found in the cycle labelling.\n" \
                 "This most likely has occurred because you are updating " \
                 "the labelling manually and not using the update() function.\n" \
                 "\nIf this error has occurred as a result of update(), please create an issue" \
                 "on github https://github.com/elykwilliams/EvasionPaths/issues"
