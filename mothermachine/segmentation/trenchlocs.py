from enum import Enum

class TrenchLocs(Enum):
    """
    simple enum to specify the trench locations to analyze
    """
    MIDDLE = 1
    TOP = 2
    BOTTOM = 3
    TOP_AND_BOTTOM = 4

