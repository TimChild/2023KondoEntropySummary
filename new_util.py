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


######################################
####### Additions to DatHDF ##########
######################################

from dat_analysis.dat.dat_hdf import FigInfo, FigInfos

    
# Monkey patch to DatHDF class
# DatHDF.standard_fig = fig_addon
# DatHDF.__repr__ = __new_dat_repr__
# DatHDF.get_Data = _get_Data
# DatHDF.save_fig = _save_fig
# DatHDF._load_fig_from_hdf = _load_fig_from_hdf
# DatHDF._load_fig_info = _load_fig_info
# DatHDF.load_fig = _load_fig
# DatHDF.saved_figs = _saved_figs
# DatHDF.plot = _plot


######################################
####### End Additions to DatHDF ######
######################################

######################################
####### Additions to Logs ############
######################################



# Monkey patch to Logs class
# Logs.scan_vars = property(scan_vars)
# Logs.sweepgates = property(get_sweepgates)






    

######################################
####### End Additions to Logs ########
######################################
    
#######################################################
########### Data class stuff  #########################
#######################################################



######################  End of Data Stuff ######################
    
    
#######################################################
########### Plotting Updates ##########################
#######################################################


def make_animated(fig: go.Figure, step_duration=0.1, copy=False, label_prefix='Datnum:'):
    """Adds animation to a slider figure

    Notes:
        Currently written for Heatmaps ONLY
        step_duration in s (converted to ms for plotly)
    """
    if copy:
        import copy

        fig = copy.deepcopy(fig)
    step_duration_ms = step_duration * 1000

    # Get the list of heatmaps and steps (TODO: Make work for scatter etc)
    heatmaps = [d for d in fig.data if isinstance(d, go.Heatmap)]
    steps = fig.layout.sliders[0].steps

    # Create Frames for animation and
    frames = []
    for step, heatmap in zip(steps, heatmaps):
        heatmap.update(visible=None)
        frames.append(
            go.Frame(
                name=step.label, data=heatmap, layout=dict(title=step.args[1]["title"])
            )
        )
        step.update(
            {
                "args": [
                    [step.label],
                    {
                        "mode": "immediate",
                    },
                ],
                "method": "animate",
            }
        )
    fig.frames = frames

    # Doesn't need all the data now, but at least 1 is necessary!
    fig.data = [heatmaps[0]]

    # Update the sliders dict
    fig.layout.sliders[0].update(
        {
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "font": {"size": 16},
                "prefix": label_prefix,
                "visible": True,
                "xanchor": "right",
            },
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
        }
    )

    # Add animate buttons to fig
    fig.update_layout(
        transition_duration=step_duration_ms,
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [
                            None,
                            {
                                "frame": {"duration": step_duration_ms, "redraw": True},
                                "fromcurrent": True,
                            },
                        ],
                        "label": "Play",
                        "method": "animate",
                    },
                    {
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                            },
                        ],
                        "label": "Pause",
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top",
            }
        ],
    )
    return fig









################# END OF PLOTTING UPDATES ################################


def clean_ct(data, jump_threshold=0.03):
    """Remove charge jumps from CT data"""
    diff = np.diff(data, axis=1)
    diff = np.concatenate(
        ([[0]] * diff.shape[0], diff), axis=1
    )  # Just add zeros to beginning to make diff have same shape as data
    diff[np.where(np.abs(diff) < jump_threshold)] = 0
    diff_sum = np.cumsum(diff, axis=1)
    return data - diff_sum
