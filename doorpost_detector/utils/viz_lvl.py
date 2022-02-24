from enum import IntEnum, auto


class VizLVL(IntEnum):
    """
    Enum for the different levels of visualisation.
    """

    NONE = auto()
    RESULT_ONLY = auto()
    EVERY_STEP = auto()
