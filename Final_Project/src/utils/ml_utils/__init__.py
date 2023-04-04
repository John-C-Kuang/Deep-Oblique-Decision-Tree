from . import text
from .text import *

framework = None
supported_frameworks = {'numpy': 'numpy', 'np': 'numpy', 'pandas': 'pandas', 'pd': 'pandas'}

# submodule imports
from .framework_handler import *

from . import metric
from .metric import *
from . import optimizer
from .optimizer import *
from . import stats
from .stats import *
from . import experimental
