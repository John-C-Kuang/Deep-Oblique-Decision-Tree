# global
import utils

# local
from utils.ml_utils import supported_frameworks


def set_framework(framework: str) -> None:
    """
    Sets the input data library handler for supported frameworks.

    @param framework: string indicator of the target framework.
    @return: None
    """
    key = framework.lower()
    if key not in supported_frameworks:
        raise RuntimeError('Given framework library unsupported')
    utils.ml_utils.framework = supported_frameworks[key]


def numpy() -> None:
    """
    Macro for setting the input data framework using NumPy.

    @return: None
    """
    set_framework('numpy')


def pandas() -> None:
    """
    Macro for setting the input data framework using Pandas

    @return: None
    """
    set_framework('pandas')
