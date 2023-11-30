import math


class AverageAggregator():
    """
    Class providing the functionality of calculating an average incrementally,
    without storing all the values explicitly.
    """

    def __init__(self, ignore_nans: bool = False):
        self._ignore_nans = ignore_nans 

        self._avg = 0
        self._count = 0

    def update(self, val: float, n: int = 1) -> None:
        """
        Update the current average.

        Args:
            val: average of an incoming sample to be included 
                in the current average.
            n: size of the incoming sample.
        """
        if self._ignore_nans and math.isnan(val):
            return

        self._avg = (
            (self._avg*self._count + val*n) / (self._count+n)
        )
        self._count += n

    def item(self) -> float:
        return self._avg