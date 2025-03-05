"""
PlotExample.py

Interface (sort of) for all example plots.
"""
from abc import ABC, abstractmethod

class PlotExample(ABC):

    @abstractmethod
    def main(self) -> None:
        """ Execute the plot example using the main function """
        pass
