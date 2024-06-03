# Description:
# Class to iterate over all possible combinations of hyperparameters.

# ================================================================================================

import itertools
from functools import reduce
from operator import mul

# ================================================================================================

class ParameterGrid:
    """Class to iterate over all possible combinations of hyperparameters.
    """
    
    def __init__(self, **kwargs):
        """Receives keyword arguments with lists of hyperparameters and calculates the total number of combinations.
        Uses the object to iterate over all possible combinations of hyperparameters.
        """
        self.kwargs_keys = kwargs.keys()
        self.kwargs_values = kwargs.values()
        self.length = len(list(itertools.product(*self.kwargs_values)))

    def __iter__(self):
        return (dict(zip(self.kwargs_keys, values)) for values in itertools.product(*self.kwargs_values))

    def __len__(self):
        return reduce(mul, (len(values) for values in self.kwargs_values), 1)
