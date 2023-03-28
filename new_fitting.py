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


class GeneralFitter(abc.ABC):
    """
    A base class for fitting data to a function
    I.e. This provides a generally useful framework and useful functions for any individual fitting
    This should be subclassed for any specific type of fitting
    
    This is intended to remain pickleable (i.e. not actually hold onto the lmfit ModelResults or references to functions etc)
    """

    def __init__(self, data: Data):
        self.data = data

        # Caching variables (to make it very fast to ask for the same fit again)
        self._last_params: lm.Parameters = None
        # Note: ModelResult is not pickleable, FitResult is (so can be passed in dash)
        self._last_fit_result: FitResult = None

    @property
    def fit_result(self) -> FitResult:
        """Nice way to access the self._last_fit_result value (i.e. equivalent to a fit result)"""
        if self._last_fit_result:
            return self._last_fit_result
        else:
            raise RuntimeError(f"No FitResult exists yet, run a fit first")

    def __repr__(self):
        return f"FitToNRG(data={repr(self.data)})"

    def _ipython_display_(self):
        return self.plot_fit()._ipython_display_()

    def fit(self, params: Optional[lm.Parameters] = _NOT_SET) -> FitInfo:
        """Do the fit, with optional parameters to have more control over max/mins/initial/vary etc"""
        if params is _NOT_SET:
            params = self._last_params  # Even if None
        if not params:
            params = self.make_params()

        # Only do expensive calculation if necessary
        if self._last_fit_result is None or self._are_params_new(params):
            self._last_params = params.copy()

            # Fit to the function
            x = self.data.x
            data = self.data.data
            model = self.model()
            fit = calculate_fit(
                x=x, data=data, params=params, func=model, method=self._default_fit_method()
            )  # Powell method works best for fitting to interpolated data

            # Cache for quicker return if same params asked again
            self._last_fit_result = FitResult.from_fit(self.data, fit.fit_result)
        return self._last_fit_result

    def eval(self, x: np.ndarray, params=_NOT_SET) -> np.ndarray:
        if params is _NOT_SET:
            params = self._last_params  # Even if None
        if not params:
            params = self.make_params()

        model = self.model()
        return model.eval(x=x, params=params)
    
    def _default_fit_method(self):
        return 'leastsq'

    @classmethod
    def model(cls):
        """Override to return model for fitting
        Examples:
            return lm.models.Model(nrg.NRG_func_generator(which='i_sense')) + lm.models.Model(simple_quadratic)
        """
        raise NotImplementedError()

    def make_params(self) -> lm.Parameters:
        """Override to return default params for fitting"""
        raise NotImplementedError

    def plot_fit(self, params: Optional[lm.Parameters] = _NOT_SET, plot_init=False):
        if params is not _NOT_SET and not are_params_equal(params, self._last_params):
            logging.warning(
                f"Plotting charge fit with different parameters to last fit"
            )

        fit_data = self.fit(params=params)
        fig = fit_data.plot(plot_init=plot_init)
        return fig

    def _are_params_new(self, params: lm.Parameters):
        if params is None:
            params = self.make_params()
        last_is_none = self._last_params is None
        pars_not_equal = not are_params_equal(params, self._last_params)
        if last_is_none or pars_not_equal:
            return True
        return False


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
    

class GeneralSimultaneousFitter:
    def __init__(
        self,
        # ftns: list[NewFitToNRG],
        datas: Union[list[Data], list[FitResult]],
    ):
        """Carry out fitting on multiple datasets simultaneously

        Args:
            datas: list of datas to fit simultaneously
                (if passing FitResult, the params there can be used for initial fits)
        """
        # Data to fit
        self.datas: Union[list[Data], list[FitResult]] = datas

        # Caching Params
        self._last_fit_result: SimultaneousFitResult = None
        self._last_params: lm.Parameters = None

    @property
    def fit_result(self) -> SimultaneousFitResult:
        """Nice way to access the self._last_fit_result value (i.e. equivalent to a fit result)"""
        if self._last_fit_result:
            return self._last_fit_result
        else:
            raise RuntimeError(f"No SimultaneousFitResult exists yet, run a fit first")
        
    def _func(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Override this with the function that should be evaluated to generate residuals
        Note: **kwargs will contain a dict[par_name: value]

        Examples:
            return nrg.nrg_func(x=x, **par_dict, data_name='i_sense')
        """
        raise NotImplementedError(
            "Need to implement self._func(...) for whatever you want to fit"
        )

    def fit(
        self,
        which: str = "charge",
        params: lm.Parameters = _NOT_SET,
    ) -> SimultaneousFitResult:
        """Fit the data simultaneosly"""
        params = self._figure_out_params(params)

        # Only do expensive calculation if necessary
        if self._last_fit_result is None or self._are_params_new(params):
            self._last_params = params

            # Collect Data to Fit
            xs, datas = self._collect_datas()

            # Do fit
            objective = self._get_objective_function()
            fit = lm.minimize(
                objective, params, method="powell", args=(xs, datas), nan_policy="omit"
            )
            self._last_fit_result = SimultaneousFitResult(
                datas=self.datas, fit=fit, plot_info=None
            )
        return self._last_fit_result

    def eval_dataset(
        self, dataset_index: int, x: np.ndarray = None, params=None, initial=False
    ) -> Data:
        """Equivalent to `eval`, but the `datset_index` also needs to be supplied to know which params to be plotting for
        Args:
            initial: Evaluate the initial fit instead
        """
        if x is None:
            x = self._last_fit_result.datas[dataset_index].x
        if params is None:
            params = self._last_fit_result.individual_params[dataset_index]
        par_dict = (
            params.valuesdict()
            if not initial
            else {k: p.init_value for k, p in params.items()}
        )
        data = self._func(x=x, **par_dict)
        return Data(
            x=x,
            data=data,
            plot_info=PlottingInfo(
                x_label="",
                y_label="",
                title=f"Dataset {dataset_index} Eval{' Initial' if initial else ''}",
            ),
        )

    def plot_fits(
        self,
        plot_init: bool = False,
        waterfall: bool = False,
        waterfall_spacing: float = None,
    ) -> go.Figure:
        fit_data = self._last_fit_result

        y_shift = 0  # For waterfall
        fig = default_fig()
        colors = pc.qualitative.D3
        for i, data in enumerate(self.datas):
            traces = []
            color = colors[i % len(colors)]
            data_trace = data.get_traces()[0]  # Only the data, no errors
            data_trace.update(
                mode="markers", name=f"data {i}", marker=dict(size=2, color=color)
            )
            traces.append(data_trace)

            fit_trace = self.eval_dataset(i).get_traces(max_num_pnts=500)[0]
            fit_trace.update(
                mode="lines",
                name=f"fit {i}",
                legendgroup=data_trace.legendgroup,
                line=dict(color=color, width=2),
            )
            traces.append(fit_trace)
            if plot_init:
                init_trace = self.eval_dataset(i, initial=True).get_traces(
                    max_num_pnts=500
                )[0]
                init_trace.update(
                    mode="lines",
                    name=f"init {i}",
                    legendgroup=data_trace.legendgroup,
                    opacity=0.6,
                    line=dict(color=color, width=2),
                )
                traces.append(init_trace)
            if waterfall:
                y_shift += (
                    waterfall_spacing
                    if waterfall_spacing
                    else 0.2 * (np.nanmax(traces[0].y) - np.nanmin(traces[0].y))
                )
                for t in traces:
                    t.update(y=t.y + y_shift)
            fig.add_traces(traces)
        return fig

    def make_params(self) -> lm.Parameters:
        params = self._make_per_dataset_params()
        combined_params = self.combine_params(params)
        return combined_params.copy()

    def update_all_param(
        self,
        params: lm.Parameters,
        param_name: str,
        value=_NOT_SET,
        vary=_NOT_SET,
        min=_NOT_SET,
        max=_NOT_SET,
    ) -> lm.Parameters:
        """Update all params that are <param_name>_i"""
        for i in range(len(self.datas)):
            for k, v in zip(["value", "vary", "min", "max"], [value, vary, min, max]):
                if v is not _NOT_SET:
                    setattr(params[f"{param_name}_{i}"], k, v)
        return params

    def make_param_shared(
        self, params: lm.Parameters, share_params: Union[str, list[str]]
    ):
        """
        Make `share_param` be shared between all simultaneous fits (i.e. vary together)
        Note: this can easily be achieved (as well as more complex relations) by modifying param.expr directly
        """
        share_params = ensure_list(share_params)
        for i in range(1, len(self.datas)):
            for share_param in share_params:
                params[f"{share_param}_{i}"].expr = f"{share_param}_0"
        return params

    def combine_params(self, params_list: list[lm.Parameters]) -> lm.Parameters:
        """Combine list of lm.Parameters into one lm.Parameters with unique names"""
        combined_params = lm.Parameters()
        for i, pars in enumerate(params_list):
            for par_name in pars.keys():
                combined_params[f"{par_name}_{i}"] = pars[par_name]
        return combined_params

    def _make_per_dataset_params(self) -> list[lm.Parameters]:
        """Make a single Parameters object containing fitting parameters for each dataset"""
        params_list = []
        for i, data in enumerate(self.datas):
            # Get initial pars
            pars = self._get_initial_dataset_params(data)

            # Add to the list of params for each dataset
            params_list.append(pars)
        return params_list

    def _figure_out_params(self, params: lm.Parameters) -> lm.Parameters:
        """Some logic about figuring out which params to use for fitting
        I.e. Use params provided if provided, else default
        """
        if params is _NOT_SET:
            params = self._last_params  # Even if None

        # If not passed, and last_params is None
        if not params:
            params = self.make_params()
        return params

    def _are_params_new(self, params: lm.Parameters) -> bool:
        if params is None:
            params = self.make_params()
        last_is_none = self._last_params is None
        pars_not_equal = not are_params_equal(params, self._last_params)
        if last_is_none or pars_not_equal:
            return True
        return False

    def _func_dataset(self, params: lm.Parameters, i: int, x: np.ndarray):
        # Collect params for this dataset (and strip off _<index> for the names
        pars = {
            re.match("(.+)_\d+", k).groups()[0]: v
            for k, v in params.items()
            if int(re.search("_(\d+)", k).groups()[0]) == i
        }
        return self._func(x=x, **pars)

    def _get_objective_function(self):
        """Objective function to minimize for all data in one go"""

        def objective(params, xs, datas):
            resids = []
            for i, (row, x) in enumerate(zip(datas, xs)):
                resids.extend(row - self._func_dataset(params, i, x))

            # 1D resids required for lm.minimize
            return resids

        return objective

    def _collect_datas(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Collect the xs and datas from each of the FitToNRG instances"""
        xs = []
        datas = []
        for data in self.datas:
            if data.x.shape[0] > 1000:
                data = data.decimate(numpnts=1000)
            xs.append(data.x.astype(np.float32))
            datas.append(data.data.astype(np.float32))
        return xs, datas

    def _separate_combined_params(
        self, combined_params: lm.Parameters
    ) -> list[lm.Parameters]:
        """Separate out the combined simultaneous fitting parameters back into the individual fitting parameters
        Note: Potentially a better way to do this is:
            - copy the full params
            - remove unwanted params
            - change name (remove _xx) from desired params
            This would keep any other useful things associated with each param that I have not considered below
        """
        params_list = separate_simultaneous_params(combined_params)
        return params_list

    def _get_initial_dataset_params(self, data: Union[Data, FitResult]):
        # if isinstance(data, FitResult) and data.params:  # TODO: switch back to this, only doesn't work with autoreload
        if hasattr(data, "params") and data.params:
            return data.params.copy()
        else:
            raise NotImplementedError(
                f"Either pass FitResult which has params or override _get_initial_dataset_params(...) to add a way to guess params from data"
            )
            
            
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
    
    
