from abc import ABC, abstractmethod

from alpha_complex import Simplex
from topology import Topology
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
    def update(self, update_data):
        ...

    ## Return cycle labelling.
    # raises key error if cycle not found.
    @abstractmethod
    def __getitem__(self, item):
        ...

    def has_intruder(self) -> bool:
        return any(self[cycle] for cycle in self)


class CycleLabellingDict(CycleLabelling):

    def __init__(self, topology: Topology):
        self.dict = dict()
        for cycle in topology.boundary_cycles:
            if Simplex(cycle.nodes) in topology.simplices(topology.dim):
                self.dict[cycle] = False
            else:
                self.dict[cycle] = True
        self.dict[topology.alpha_cycle] = False

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

    ## Return cycle labelling.
    # raises key error if cycle not found.
    def __setitem__(self, item, value):
        if item not in self.dict:
            raise LabellingError("Item does not have a label")
        self.dict[item] = value

    # Given an UpdateData object, do the following
    #   add all new cycles,
    #   update all labels
    #   remove all old cycles
    def update(self, update_data):
        if not self.is_valid(update_data):
            raise LabellingError("Invalid update provided to labelling")

        for cycle in update_data.cycles_added:
            self.dict[cycle] = True
        self.dict.update(update_data.mapping)
        for cycle in update_data.cycles_removed:
            del self.dict[cycle]

    def is_valid(self, update_data):
        if any(cycle not in self for cycle in update_data.cycles_removed):
            return False
        elif not all(cycle in self or cycle in update_data.cycles_added for cycle in update_data.mapping):
            return False
        else:
            return True
