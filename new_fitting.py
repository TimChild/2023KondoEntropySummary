"""
2023-03-28 -- All useful functions moved to dat_analysis (and imported here for backwards-compatability)
"""
from __future__ import annotations
from typing import Callable, Iterable, Optional, Union
import plotly.graph_objects as go
import plotly.colors as pc
import numpy as np
import pandas as pd
import lmfit as lm
import logging
import re
import abc
from dataclasses import field, dataclass, InitVar


from dat_analysis.analysis_tools import nrg
from dat_analysis.analysis_tools.general_fitting import (
    calculate_fit,
    separate_simultaneous_params,
    FitInfo,
)
from dat_analysis.useful_functions import ensure_list
from dat_analysis.plotting.plotly.util import default_fig
from new_util import (
    InterlacedData,
    Data,
    PlottingInfo,
    FitResult,
    are_params_equal,
    SimultaneousFitResult,
)

from dat_analysis.analysis_tools.general_fitting import (
    GeneralFitter,
    GeneralSimultaneousFitter,
)
from dat_analysis.analysis_tools.nrg import (
    NRGChargeFitter,
    NRGConductanceFitter,
    NRGChargeSimultaneousFitter,
    NRGConductanceSimultaneousFitter,
    NRGEntropySignalFitter,
    _simple_quadratic as simple_quadratic,
)
