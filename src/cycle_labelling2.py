from abc import ABC, abstractmethod

from topology2 import Topology
from update_data import LabelUpdate


class AbstractCycleLabelling(ABC):

    ## Check if cycle has label.
    @abstractmethod
    def __contains__(self, item):
        ...

    ## Iterate through all cycles.
    @abstractmethod
    def __iter__(self):
        ...

    ## Return cycle labelling.
    # raises key error if cycle not found.
    @abstractmethod
    def __getitem__(self, item):
        ...

    ## Set cycle label.
    # raises KeyError if not found
    @abstractmethod
    def __setitem__(self, key, value):
        ...

    ## Remove cycle from tree.
    @abstractmethod
    def __delitem__(self, key):
        ...

    # Given an UpdateData object, do the following
    #   add all new cycles,
    #   update all labels
    #   remove all old cycles
    @abstractmethod
    def update(self, update_data: LabelUpdate):
        ...


class CycleLabellingDict(AbstractCycleLabelling):

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

    ## Set cycle label.
    # raises KeyError if not found
    def __setitem__(self, key, value):
        if key not in self.dict:
            raise LabellingError("Item does not have a label")
        self.dict[key] = value

    ## Remove cycle from tree.
    def __delitem__(self, key):
        del self.dict[key]

    # Given an UpdateData object, do the following
    #   add all new cycles,
    #   update all labels
    #   remove all old cycles
    def update(self, update_data: LabelUpdate):
        self.dict.update(update_data.mapping)
