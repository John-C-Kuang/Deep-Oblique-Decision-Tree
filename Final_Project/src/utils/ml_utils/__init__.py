framework = None
supported_frameworks = {'numpy': 'numpy', 'np': 'numpy', 'pandas': 'pandas', 'pd': 'pandas'}

# submodule imports
from .framework_handler import *

from . import metric
from .metric import *
from . import experimental
