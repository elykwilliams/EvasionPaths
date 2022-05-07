# ************************************************************
# Copyright (c) 2021, Kyle Williams - All Rights Reserved.
# You may use, distribute and modify this code under the
# terms of the BSD-3 license. You should have received a copy
# of the BSD-3 license with this file.
# If not, visit: https://opensource.org/licenses/BSD-3-Clause
# ************************************************************


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
        self._cycle_label = dict()

        for cycle in state.boundary_cycles():
            self._cycle_label[cycle] = True
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




    #################### 3D Atomic Changes ############################

    # (1, 0, 0, 0, 0, 0, 0, 0)  is not included because the boundary cycles keep their labels

    # (0, 1, 0, 0, 0, 0, 0, 0) is not included because the boundary cycles keep the labeling false

    # (0, 0, 0, 0, 1, 0, 0, 0)

    def _add_3simplex(self, added_simplices):
        for simplex in added_simplices:
            self._cycle_label[simplex] = False

    # (0, 0, 0, 0, 0, 1, 0, 0)

    # Question here, what about when the sensor network is first initialized? Do tetrahedrons get marked as false?

    # def _remove_3simplex(self, removed_simplices):
    #     for simplex in removed_simplices:
    #         self._cycle_label[simplex] = False

    # (0, 0, 1, 0, 0, 0, 1, 1)
    def _add_2simplex_1to1(self, removed_cycles, added_cycles):
        # Not done
        self._cycle_label[added_cycles[0]] = self._cycle_label[removed_cycles]
        self._delete_all(removed_cycles)

    # (0, 0, 0, 1, 0, 0, 1, 1)
    def _remove_2simplex_1to1(self, removed_cycles, added_cycles):
        # Not done
        self._cycle_label[added_cycles[0]] = self._cycle_label[removed_cycles]
        self._delete_all(removed_cycles)

    # (0, 0, 1, 0, 0, 0, 2, 1)
    def _add_2simplex_2to1(self, removed_cycles, added_cycles):
        for cycle in added_cycles:
            self._cycle_label[cycle] = self._cycle_label[removed_cycles[0]]
        self._delete_all(removed_cycles)

    # (0, 0, 0, 1, 0, 0, 1, 2)
    def _remove_2simplex_1to2(self, removed_cycles, added_cycles):
        self._cycle_label[added_cycles[0]] = any([self._cycle_label[s] for s in removed_cycles])
        self._delete_all(removed_cycles)

    # (1, 0, 1, 0, 0, 0, 1, 1)
    def _add_simplex_pair(self, removed_cycles, added_cycles):
        # Not done
        self._cycle_label[added_cycles[0]] = self._cycle_label[removed_cycles]
        self._delete_all(removed_cycles)

    # (0, 1, 0, 1, 0, 0, 1, 1)
    def _remove_simplex_pair(self, removed_cycles, added_cycles):
        # Not done
        self._cycle_label[added_cycles[0]] = self._cycle_label[removed_cycles]
        self._delete_all(removed_cycles)

    # (0, 0, 1, 0, 1, 0, 2, 1)
    def _fill_3simplex_face(self):
        # Not done
        return True

    # (0, 0, 0, 1, 0, 1, 1, 2)
    def _remove_3simplex_face(self, removed_cycles, added_cycles):
        self._cycle_label[added_cycles[0]] = any([self._cycle_label[s] for s in removed_cycles])
        self._delete_all(removed_cycles)

    # (1, 0, 2, 0, 1, 0, 2, 1)
    def _fill_tetrahedron_edge(self, removed_cycles, added_cycles):
        # Not done
        return True

    # (0, 1, 0, 2, 0, 1, 1, 2)
    def _remove_tetrahedron_edge(self, removed_cycles, added_cycles):
        self._cycle_label[added_cycles[0]] = any([self._cycle_label[s] for s in removed_cycles])
        self._delete_all(removed_cycles)

    # (1, 0, 3, 1, 3, 2, 3, 2)
    def _3Ddelaunay_projection_3to2(self, removed_cycles, added_cycles):
        # Not done
        return True

    # (0, 1, 1, 3, 2, 3, 2, 3)
    def _3Ddelaunay_projection_2to3(self, removed_cycles, added_cycles):
        # Not done
        return True



    ################### 2D Code for Reference #######################

    # ## Add edge.
    # # When an edge is added, one boundary cycle is split into two
    # # the two new boundary cycles are added, and the old boundary
    # # cycle is removed.
    # # Note that both parameters should be lists of boundary cycles.
    #
    # def _add_1simplex(self, removed_cycles, added_cycles):
    #     for cycle in added_cycles:
    #         self._cycle_label[cycle] = self._cycle_label[removed_cycles[0]]
    #     self._delete_all(removed_cycles)
    #
    # ## Remove edge.
    # # When an edge is removed, two boundary cycles join into one.
    # # The two old boundary cycles are removed, and the new boundary
    # # cycle is added.
    # # Note that both parameters should be lists of boundary cycles.
    # def _remove_1simplex(self, removed_cycles, added_cycles):
    #     self._cycle_label[added_cycles[0]] = any([self._cycle_label[s] for s in removed_cycles])
    #     self._delete_all(removed_cycles)
    #
    # ## Mark boundary cycle as a 2-simplex.
    # # added_simplices should be a list of boundary cycles corresponding to
    # # the new 2-simplices
    # def _add_2simplex(self, added_simplices):
    #     for simplex in added_simplices:
    #         self._cycle_label[simplex] = False
    #
    # ## Add 2-simplex + edge.
    # # Need to add edge first so that added simple is in labelling.
    # # removed_cycles and added_cycles should be lists of boundary cycles
    # # added_simplex is the boundary cycle of the added simplex.
    # def _add_simplex_pair(self, removed_cycles, added_cycles, added_simplices):
    #     self._add_1simplex(removed_cycles, added_cycles)
    #     self._add_2simplex(added_simplices)
    #
    # ## Remove edge and 2-simplex.
    # # This is the same logic as removing just an edge.
    # # removed_cycles and added_cycles should be lists of boundary cycles
    # def _remove_simplex_pair(self, removed_cycles, added_cycles):
    #     self._remove_1simplex(removed_cycles, added_cycles)
    #
    # ## Delauny filp.
    # # add 2 new simplices, remove two old simplices.
    # # removed_cycles and added_cycles should be lists of boundary cycles
    # def _delaunay_flip(self, removed_cycles, added_cycles):
    #     self._add_2simplex(added_cycles)
    #     self._delete_all(removed_cycles)

    #################### End 2D code #########################


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


        ####################### 3D code #####################

        # Add a 1-simplex
        elif state_change.case == (1, 0, 0, 0, 0, 0, 0, 0):
            return
        # Remove a 1-simplex
        elif state_change.case == (0, 1, 0, 0, 0, 0, 0, 0):
            return

        # 3-simplex is added so we need to find these and mark the boundary
        # cycles and label them as false.
        elif state_change.case == (0, 0, 0, 0, 1, 0, 0, 0):
            simplex = state_change.simplices_added[0]
            added_simplices = [state_change.new_state.simplex2cycle(simplex)]
            self._add_3simplex(added_simplices)

        # 3-simplex is removed so we keep whatever the previous labeling was.

        elif state_change.case == (0, 0, 0, 0, 0, 1, 0, 0):
            return



        # Check these two later.
        # The new boundary cycles inherit the original labeling of the removed ones.
        elif state_change.case == (0, 0, 1, 0, 0, 0, 1, 1):
            self._add_2simplex_1to1(state_change.cycles_removed, state_change.cycles_added)

        elif state_change.case == (0, 0, 0, 1, 0, 0, 1, 1):
            self._remove_2simplex_1to1(state_change.cycles_removed, state_change.cycles_added)


        # 2-simplex is added which splits the boundary cycles and they obtain the old labeling.

        elif state_change.case == (0, 0, 1, 0, 0, 0, 2, 1):
            self._add_2simplex_2to1(state_change.cycles_removed, state_change.cycles_added)

        # 2-simplex is removed so we need to check both of the old boundary cycles
        # to see if it was possible to contain an intruder.

        elif state_change.case == (0, 0, 0, 1, 0, 0, 1, 2):
            self._remove_2simplex_1to2(state_change.cycles_removed, state_change.cycles_added)


        # Check these next two later

        elif state_change.case == (1, 0, 1, 0, 0, 0, 1, 1):
            self._add_simplex_pair(state_change.cycles_removed, state_change.cycles_added)

        elif state_change.case == (0, 1, 0, 1, 0, 0, 1, 1):
            self._remove_simplex_pair(state_change.cycles_removed, state_change.cycles_added)



        # Face fills in the tetrahedron so we get rid of the old boundary cycle which
        # went into the hold and it gets split into two. We then must label the boundary
        # cycle on the 3 simplex as false since it gets filled in and the outer boundary
        # cycle gets the previous label.

        elif state_change.case == (0, 0, 1, 0, 1, 0, 2, 1):
            # not done, need to find the 3-simplex.
            return

        # tetrahedron has a face split open and thus a cavity is created.
        # We must then inherit the labeling of the outer boundary cycle that
        # corresponded to the outside. We can also just use an Or labeling to
        # avoid the search (it might just be faster if we had a much larger problem).

        elif state_change.case == (0, 0, 0, 1, 0, 1, 1, 2):
            self._remove_3simplex_face(state_change.cycles_removed, state_change.cycles_added)

        # Edge is added which completes the tetrahedron and thus completing the two faces.
        # The outer boundary cycle inherits the labeling of the old boundary cycle
        # and the boundary cycle pertaining to the 3-simplex is labeled as false.

        elif state_change.case == (1, 0, 2, 0, 1, 0, 2, 1):
            # Not done
            return

        # The tetrahedron opens up and the two old boundary cycles become one.
        # We can use an OR operator for this.
        elif state_change.case == (0, 1, 0, 2, 0, 1, 1, 2):
            self._remove_tetrahedron_edge(state_change.cycles_removed, state_change.cycles_added)

        # The next two cases are a kind of delaunay flip in 3D.
        elif state_change.case == (1, 0, 3, 1, 3, 2, 3, 2):
            # not done
            return

        elif state_change.case == (0, 1, 1, 3, 2, 3, 2, 3):
            # not done
            return



        #################### 2D code for the cases #########################
        # Add 1-Simplex
        elif state_change.case == (1, 0, 0, 0, 2, 1):
            self._add_1simplex(state_change.cycles_removed, state_change.cycles_added)

        # Remove 1-Simplex
        elif state_change.case == (0, 1, 0, 0, 1, 2):
            self._remove_1simplex(state_change.cycles_removed, state_change.cycles_added)

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


# 2D code, not sure what the equivalent of this is in 3D.
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

