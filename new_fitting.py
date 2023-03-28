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
from dat_analysis.analysis_tools.general_fitting import calculate_fit, separate_simultaneous_params, FitInfo
from dat_analysis.useful_functions import ensure_list
from dat_analysis.plotting.plotly.util import default_fig
from new_util import InterlacedData, Data, PlottingInfo, FitResult, are_params_equal, SimultaneousFitResult

# For being able to check whether None is intentionally passed in
_NOT_SET = object()


def simple_quadratic(x, quad):
    """For adding a quadratic contribution to existing model"""
    # Note: 1e-10 is just to help lmfit not have to use such tiny values for quad term
    return 1e-10*quad*x**2  


from dat_analysis.analysis_tools.general_fitting import GeneralFitter, GeneralSimultaneousFitter



class NRGChargeFitter(GeneralFitter):
    def _default_fit_method(self):
        return 'powell'
    
    @classmethod
    def model(cls) -> lm.models.Model:
        return lm.models.Model(
            nrg.NRG_func_generator(which="i_sense")
        ) + lm.models.Model(simple_quadratic)

    def make_params(self) -> lm.Parameters:
        x = self.data.x
        data = self.data.data

        first_non_nan_index = pd.Series(data).first_valid_index()
        last_non_nan_index = pd.Series(data).last_valid_index()

        theta = 50
        gamma = 10
        lin = (data[last_non_nan_index] - data[first_non_nan_index]) / (
            x[last_non_nan_index] - x[first_non_nan_index]
        )
        amp = np.nanmax(data - x * lin) - np.nanmin(data - x * lin)
        occ_lin = 0
        const = np.nanmean(data)

        # Create lm.parameters with some initial values and limits
        params = lm.Parameters()
        params.add_many(
            ("mid", 0, True, np.nanmin(x), np.nanmax(x), None, None),
            ("theta", theta, False, 0.5, 1000, None, None),
            # Limit to Range of NRG G/Ts
            ("g", gamma, True, theta / 1000, theta * 50, None, None),
            ("amp", amp, True, 0, 3, None, None),
            ("const", const, True, None, None, None, None),
            ("lin", lin, True, -0.005, 0.005, None, None),
            ("quad", 0, False, None, None, None, None),
            ("occ_lin", occ_lin, False, None, None, None, None),
        )
        return params

    def plot_fit(
        self,
        params: Optional[lm.Parameters] = _NOT_SET,
        plot_init=False,
        sub_linear=False,
    ):
        fig = super().plot_fit(params=params, plot_init=plot_init)
        fit = self.fit(params=params)
        fig.update_layout(title="Charge Fit", yaxis_title="Current /nA")
        if sub_linear:
            m = fit.params.best_values.lin
            c = fit.params.best_values.const
            for data in fig.data:
                data.y = data.y - (m * data.x + c)
            fig.update_layout(title=fig.layout.title.text + " sub linear")
        return fig


class NRGConductanceFitter(GeneralFitter):
    def _default_fit_method(self):
        return 'powell'
    
    @classmethod
    def model(cls) -> lm.models.Model:
        return lm.models.Model(nrg.NRG_func_generator(which="conductance"))

    def make_params(self) -> lm.Parameters:
        x = self.data.x
        data = self.data.data

        theta = 50
        gamma = 10
        amp = 1
        const = 0

        # Create lm.parameters with some initial values and limits
        params = lm.Parameters()
        params.add_many(
            ("mid", 0, True, np.nanmin(x), np.nanmax(x), None, None),
            ("theta", theta, True, 0.5, 1000, None, None),
            # Limit to Range of NRG G/Ts
            ("g", gamma, True, theta / 1000, theta * 50, None, None),
            # Amplitude of Data unlikely to match NRG
            ("amp", amp, True, 0, None, None, None),
            # May want to allow for offset, but default to
            ("const", const, False, None, None, None, None),
            # Unused parameters required to stop nrg function complaining
            ("lin", 0, False, None, None, None, None),
            ("occ_lin", 0, False, None, None, None, None),
        )
        return params

    def plot_fit(
        self,
        params: Optional[lm.Parameters] = _NOT_SET,
        plot_init=False,
        sub_linear=False,
    ):
        fig = super().plot_fit(params=params, plot_init=plot_init)
        fit = self.fit(params=params)
        fig.update_layout(title="Conductance Fit")
        return fig

    
class NRGEntropySignalFitter(GeneralFitter):
    def _default_fit_method(self):
        return 'powell'
    
    @classmethod
    def model(cls) -> lm.models.Model:
        return lm.models.Model(
            nrg.NRG_func_generator(which="dndt")
        )

    def make_params(self) -> lm.Parameters:
        x = self.data.x
        data = self.data.data

        first_non_nan_index = pd.Series(data).first_valid_index()
        last_non_nan_index = pd.Series(data).last_valid_index()

        theta = 50
        gamma = 10
        lin = (data[last_non_nan_index] - data[first_non_nan_index]) / (
            x[last_non_nan_index] - x[first_non_nan_index]
        )
        amp = np.nanmax(data) - np.nanmin(data)

        # Create lm.parameters with some initial values and limits
        params = lm.Parameters()
        params.add_many(
            ("mid", 0, True, np.nanmin(x), np.nanmax(x), None, None),
            ("theta", theta, False, 0.5, 1000, None, None),
            # Limit to Range of NRG G/Ts
            ("g", gamma, True, theta / 1000, theta * 50, None, None),
            ("amp", amp, True, 0, None, None, None),
            ("const", 0, False, None, None, None, None),
            ("lin", 0, False, None, None, None, None),
            # ("quad", 0, False, None, None, None, None),
            ("occ_lin", 0, False, None, None, None, None),
        )
        return params

    def plot_fit(
        self,
        params: Optional[lm.Parameters] = _NOT_SET,
        plot_init=False,
        sub_linear=False,
    ):
        fig = super().plot_fit(params=params, plot_init=plot_init)
        fit = self.fit(params=params)
        fig.update_layout(title="dN/dT Fit", yaxis_title="Delta Current /nA")
        return fig
    

class NRGChargeSimultaneousFitter(GeneralSimultaneousFitter):
    def _func(self, x, **kwargs) -> np.ndarray:
        """Function to be fit (this is called to generate guess that is subtracted from data to make residuals)"""
        quad = kwargs.pop("quad", 0)
        data = nrg.nrg_func(x=x, **kwargs, data_name="i_sense")
        if quad != 0:
            data += simple_quadratic(x, quad)
        return data


class NRGConductanceSimultaneousFitter(GeneralSimultaneousFitter):
    def _func(self, x, **kwargs) -> np.ndarray:
        """Function to be fit (this is called to generate guess that is subtracted from data to make residuals)"""
        data = nrg.nrg_func(x=x, **kwargs, data_name="conductance")
        return data
    
    
