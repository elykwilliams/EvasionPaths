from utilities import *


## Error to be used when checking for discrepancies between existing labelling and requested update
class UpdateError(Exception):
    pass


class UpdateData:
    def __init__(self, tree_labels, state_change):
        if not state_change.is_atomic():
            raise UpdateError("The attempted state-change is non-atomic")
        self.labels = tree_labels
        self.cycles_added = state_change.cycles_added
        self.cycles_removed = state_change.cycles_removed
        self.simplices_added = state_change.simplices_added
        self.simplices_removed = state_change.simplices_removed

        try:
            self.is_valid()
        except UpdateError:
            raise

    def is_valid(self):
        pass

    @property
    def label_update(self):
        return dict()


class Add2Simplices(UpdateData):

    @property
    def label_update(self):
        return {cycle: False for cycle in self.simplices_added}

    def is_valid(self):
        if any(cycle not in self.labels for cycle in self.simplices_added):
            raise UpdateError(f"You are attempting to add a simplex that has not yet been added to the tree")


class Remove2Simplices(UpdateData):

    def is_valid(self):
        if any(cycle not in self.labels for cycle in self.simplices_removed):
            raise UpdateError(f"You are attempting to remove a simplex that has not yet been added to the tree")


class Add1Simplex(UpdateData):
    ## Add edge.
    # added cycle will have intruder if either removed cycle has intruder
    @property
    def label_update(self):
        return {cycle: self.labels[self.cycles_removed[0]] for cycle in self.cycles_added}

    # removed cycles list should have only 1 or 2 cycles
    # cycles_added should be a list with only 1 cycle
    # This should only ever result in one cycle being removed
    def is_valid(self):
        if self.cycles_removed[0] not in self.labels:
            raise UpdateError("You are attempting to remove a boundary cycle that was never given a labelling")
        if len(self.cycles_removed) != 1:
            raise UpdateError("Adding edge cannot cause more than 1 boundary cycle to be removed")


class Remove1Simplex(UpdateData):
    ## Remove edge.
    @property
    def label_update(self):
        return {cycle: any([self.labels[cycle] for cycle in self.cycles_removed]) for cycle in self.cycles_added}

    # This should only ever result in 1 cycle being added
    def is_valid(self):
        if len(self.cycles_added) != 1:
            raise UpdateError("Adding edge cannot cause more than 1 boundary cycle to be removed")


class AddSimplexPair(UpdateData):
    ## Add simplex pair
    # Will split a cycle into two with appropriate label, then label 2simplex as False
    @property
    def label_update(self):
        return {cycle: False if cycle in self.simplices_added else self.labels[self.cycles_removed[0]]
                for cycle in self.cycles_added}

    def is_valid(self):
        if self.cycles_removed[0] not in self.labels:
            raise UpdateError("You are attempting to remove a boundary cycle that has not been given a labelling")
        if not is_subset(self.simplices_added, self.cycles_added):
            raise UpdateError("You are attempting to add a simplex that is not also being added as a boundary cycle")
        if len(self.cycles_removed) != 1:
            raise UpdateError("You are attempting to remove too many simplices at once.")
        if len(self.cycles_added) != 2:
            raise UpdateError("You are not removing two boundary cycles, this is not possible")


class RemoveSimplexPair(Remove1Simplex):

    def is_valid(self):
        super().is_valid()
        if len(self.cycles_removed) != 2:
            raise UpdateError("You are not removing two boundary cycles, this is not possible")


class DelaunyFlip(UpdateData):
    ## Delauny Flip.
    # edge between two simplices flips resulting in two new simplices
    # Note that this can only happen with simplices.
    @property
    def label_update(self):
        return {cycle: False for cycle in self.simplices_added}

    def is_valid(self):
        if len(self.simplices_removed) != 2 or len(self.simplices_added) != 2:
            raise UpdateError("You are attempting to do a delauny flip with more(or fewer) than two 2-simplices")


## Detemine label updates.
# return dictionary cycles cycles that need updating and their new label
def get_update_data(labelling, state_change):
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
