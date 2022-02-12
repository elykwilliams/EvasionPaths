from abc import ABC, abstractmethod

from topology2 import Topology
from update_data import LabelUpdate
from utilities import LabellingError


class CycleLabelling(ABC):

    ## Check if cycle has label.
    @abstractmethod
    def __contains__(self, item):
        ...

    ## Iterate through all cycles.
    @abstractmethod
    def __iter__(self):
        ...

    # Given an UpdateData object, do the following
    #   add all new cycles,
    #   update all labels
    #   remove all old cycles
    @abstractmethod
    def update(self, update_data: LabelUpdate):
        ...


class CycleLabellingDict(CycleLabelling):

    def __init__(self, topology: Topology):
        self.dict = dict()
        for cycle in topology.boundary_cycles:
            self.dict[cycle] = True

        simplex_cycles = [simplex.to_cycle(topology.boundary_cycles) for simplex in topology.simplices(2)]
        for cycle in simplex_cycles:
            self.dict[cycle] = False

    ## Check if cycle has label.
    def __contains__(self, item):
        return item in self.dict

    ## Iterate through all cycles.
    def __iter__(self):
        return iter(self.dict)

    ## Return cycle labelling.
    # raises key error if cycle not found.
    def __getitem__(self, item):
        if item not in self.dict:
            raise LabellingError("Item does not have a label")
        return self.dict[item]

    # Given an UpdateData object, do the following
    #   add all new cycles,
    #   update all labels
    #   remove all old cycles
    def update(self, update_data: LabelUpdate):
        if not self.is_valid(update_data):
            raise LabellingError("Invalid update provided to labelling")

        for cycle in update_data.cycles_added:
            self.dict[cycle] = True
        self.dict.update(update_data.mapping)
        for cycle in update_data.cycles_removed:
            del self.dict[cycle]

    def is_valid(self, update_data: LabelUpdate):
        if any(cycle not in self for cycle in update_data.cycles_removed):
            return False
        elif not all(cycle in self or cycle in update_data.cycles_added for cycle in update_data.mapping):
            return False
        else:
            return True
