"""For any general utility functions created during this experiment

Intention is to put these all into the dat_analysis package at the end of the CD
"""
from __future__ import annotations
from dataclasses import dataclass, field, InitVar
import numpy as np
import plotly.graph_objects as go
import lmfit as lm
from scipy.signal import savgol_filter
import copy
import numbers
import json
import uuid
import pandas as pd
import re
import h5py
from scipy.signal import filtfilt, iirnotch
import logging 
from typing import Any, Union, Optional
import plotly.io as pio
import unicodedata
import datetime

from dat_analysis.dat.dat_hdf import DatHDF 
from dat_analysis.dat.logs_attr import Logs
from dat_analysis.plotting.plotly.util import default_fig, heatmap, error_fill, figures_to_subplots, fig_waterfall, limit_max_datasize
from dat_analysis.useful_functions import get_matching_x, bin_data, decimate, center_data, mean_data, resample_data, get_data_index, ensure_list
from dat_analysis.analysis_tools.data_aligning import subtract_data, subtract_data_1d
from dat_analysis.hdf_util import NotFoundInHdfError

from dat_analysis.analysis_tools.data import Data, InterlacedData, PlottingInfo
from dat_analysis.analysis_tools.general_fitting import are_params_equal, FitResult, SimultaneousFitResult
from dat_analysis.core_util import slugify
from dat_analysis.plotting.plotly.util import error_fill
from dat_analysis.dat.logs_attr import AxisGates, SweepGates, convert_sweepgates_to_real
from dat_analysis.dat.dat_hdf import FigInfo, FigInfos

from dat_analysis.plotting.plotly.util import make_animated


def clean_ct(data, jump_threshold=0.03):
    """Remove charge jumps from CT data"""
    diff = np.diff(data, axis=1)
    diff = np.concatenate(
        ([[0]] * diff.shape[0], diff), axis=1
    )  # Just add zeros to beginning to make diff have same shape as data
    diff[np.where(np.abs(diff) < jump_threshold)] = 0
    diff_sum = np.cumsum(diff, axis=1)
    return data - diff_sum
