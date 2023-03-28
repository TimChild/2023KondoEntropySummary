"""
2023-03-28 -- All useful functions moved to dat_analysis (and imported here for backwards-compatability)

2023-03-28 -- Note: the remaining classes and functions in here have already been mostly superseded by more
general functions that can now be found in dat_analysis.analysis_tools.nrg. Some of the combined conductance/charge
stuff here is unique to this file, but if we want to use it again, it should be re-written with some more thought put
into keeping it general.
"""
from __future__ import annotations
from dataclasses import dataclass
import lmfit as lm
import numpy as np
import copy
import re
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, TYPE_CHECKING
import logging
from scipy.interpolate import interp1d

from dat_analysis.analysis_tools.general_fitting import FitInfo, calculate_fit
from dat_analysis.analysis_tools import nrg
import dat_analysis.useful_functions as U
from dat_analysis.plotting.plotly.util import default_fig, heatmap, figures_to_subplots, default_layout, error_fill
from new_util import Data, PlottingInfo, are_params_equal
from peak_fitting import PeakFitting


_NOT_SET = object()


def simple_quadratic(x, quad):
    """For adding a quadratic contribution to existing model"""
    return quad*x**2


class CombinedDataAveraging:
    """
    A class used to center/average conductance and charge transition data such that they share an x-axis and alignment
    """
    def __init__(self, conductance: Data, charge: Data, peak_loc: float = None, peak_width: float=None):
        self.conductance = conductance  # 2D conductance data (x, y, data)
        self.charge = (
            charge  # 2D charge sense data (x, y, data) -- could be different shape
        )
        self.peak_loc = peak_loc
        self.peak_width=peak_width

        # For internal calculation
        self.peak_fitting = PeakFitting(self.conductance, peak_loc=self.peak_loc, peak_width=self.peak_width)

        # Variables for caching
        self._centers = None

    @property
    def avg_conductance(self) -> Data:
        return self._get_avg_data(which="conductance")

    @property
    def avg_charge(self) -> Data:
        return self._get_avg_data(which="charge")

    def __repr__(self):
        return (
            f"CombinedDataAveraging(conductance=Data(x.shape = {self.conductance.x.shape}, y.shape = {self.conductance.y.shape}, data.shape = {self.conductance.data.shape}),"
            + f"charge=Data(x.shape = {self.charge.x.shape}, y.shape = {self.charge.y.shape}, data.shape = {self.charge.data.shape}))"
        )
    
    def _get_centers(self):
        """Get centers from conductance data"""
        if self._centers is None:
            centers = self.peak_fitting.get_centers()
            self._centers = centers
        return self._centers

    def _get_avg_data(self, which: str) -> Data:
        centers = self._get_centers()
        if which == "conductance":
            data = self.conductance
        elif which == "charge":
            data = self.charge
        else:
            raise NotImplementedError(f"{which} not recognized")
        avg_data, avg_x, std = U.mean_data(
            x=data.x, data=data.data, centers=centers, return_x=True, return_std=True
        )
        return Data(data=avg_data, x=avg_x, yerr=std)
    
    def plot_2d_data(self, max_pts_x=300, max_pts_y=100, resample_method="bin"):
        figs = []
        for data, title, ylabel in zip(
            [self.conductance, self.charge], ["Conductance", "Charge Transition"], ['Conductance /2e^2/h', 'Current /nA']
                                                                                    
        ):
            x = data.x
            y = data.y
            data = data.data
            # resample in x-direction
            data, x = U.resample_data(
                data,
                x=x,
                max_num_pnts=max_pts_x,
                resample_x_only=True,
                resample_method=resample_method,
            )
            # resample in y-direction
            data, y = U.resample_data(
                data.T,
                x=y,
                max_num_pnts=max_pts_y,
                resample_x_only=True,
                resample_method=resample_method,
            )
            data = data.T
            fig = default_fig()
            fig.add_trace(heatmap(x=x, y=y, data=data))
            fig.update_layout(title=title, yaxis_title=ylabel)
            figs.append(fig)
        fig = figures_to_subplots(figs, cols=2, title=f"2D Data to Average")
        return fig
    
    def plot_aligned_data(self, num_peak_fits=1):
        fig_aligned = self.peak_fitting.plot_aligned()
        fig_peak_fits = self.peak_fitting.plot_fits()
        fig = figures_to_subplots(
            [fig_aligned, fig_peak_fits], cols=2, title="Aligned Conductance"
        )
        return fig

    def plot_average_data(self, sub_linear=True):
        avg_conductance = self.avg_conductance
        avg_charge = self.avg_charge

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.update_layout(**default_layout)
        for data, secondary, name in zip(
            [avg_conductance, avg_charge],
            [False, True],
            ["Conductance", "Charge Transition"],
        ):
            x, data, yerr = data.x, data.data, data.yerr
            if name == "Charge Transition" and sub_linear:
                line = lm.models.LinearModel()
                fit = line.fit(data, x=x, nan_policy="omit")
                data = data - fit.eval(x=x)
            fig.add_trace(go.Scatter(x=x, y=data, name=name), secondary_y=secondary)
            fig.add_trace(error_fill(x=x, data=data, error=yerr), secondary_y=secondary)
        fig.update_layout(
            title=f'Averaged Conductance and Charge Transition data {"(sub linear)" if sub_linear else ""}',
            yaxis_title="Conductance /2e^2/h",
            yaxis2_title="Current /nA",
        )
        return fig

    
class CombinedData:
    """
    A class to hold the combined conductance and ct data that will be worked on for fitting/comparing to NRG etc
    """

    def __init__(self, conductance: Data = None, charge: Data = None):
        if conductance is not None and charge is not None and (conductance.data.ndim != 1 or charge.data.ndim != 1):
            raise ValueError(
                f"Both conductance and charge data must be 1D. Use CombinedDataAveraging(...) first to get averaged data"
            )
         # 1D conductance data (x, y, data)
        self.conductance = conductance if conductance is not None else Data(x=np.array([]), data=np.array([]))
        # 1D charge sense data (x, y, data) -- could be different shape
        self.charge = charge if charge is not None else Data(x=np.array([]), data=np.array([]))
    

    def __repr__(self):
        return (
            f"CombinedDataAveraging(conductance=Data(x.shape = {self.conductance.x.shape}, data.shape = {self.conductance.data.shape}),"
            + f"charge=Data(x.shape = {self.charge.x.shape}, data.shape = {self.charge.data.shape}))"
        )
    
    def plot_data(self, sub_linear=True):
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.update_layout(**default_layout)
        for data, secondary, name in zip(
            [self.conductance, self.charge],
            [False, True],
            ["Conductance", "Charge Transition"],
        ):
            x, data, yerr = data.x, data.data, data.yerr
            if name == "Charge Transition" and sub_linear:
                line = lm.models.LinearModel()
                fit = line.fit(data, x=x, nan_policy="omit")
                data = data - fit.eval(x=x)
            fig.add_trace(go.Scatter(x=x, y=data, name=name), secondary_y=secondary)
            fig.add_trace(error_fill(x=x, data=data, error=yerr), secondary_y=secondary)
        fig.update_layout(
            title=f'Conductance and Charge Transition data {"(sub linear)" if sub_linear else ""}',
            yaxis_title="Conductance /2e^2/h",
            yaxis2_title="Current /nA",
        )
        return fig


class FitToNRG:
    """
    A class for fitting conductance/CT data to NRG to determine fit parameters for later comparison
    """
    def __init__(self, combined_data: CombinedData, theta: float = None):
        self.combined_data = combined_data

        self.theta = theta

        # Caching variables
        # TODO: Are these last_params actually working (I might be storing them AFTER fitting, which will be different to before fitting...)
        self._last_charge_params = None
        self._last_charge_fit = None
        self._last_conductance_params = None
        self._last_conductance_fit = None

    @property
    def nrg_charge_func(self):
        return nrg.NRG_func_generator(which="i_sense")
    
    @property
    def nrg_conductance_func(self):
        return nrg.NRG_func_generator(which="conductance")
    
    def __repr__(self):
        return f"FitToNRG(combined_data={repr(self.combined_data)}, theta={self.theta})"
    
    def make_charge_params(self) -> lm.Parameters:
        x = self.combined_data.charge.x
        data = self.combined_data.charge.data

        first_non_nan_index = pd.Series(data).first_valid_index()
        last_non_nan_index = pd.Series(data).last_valid_index()

        theta = self.theta
        gamma = 0.01
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
            ("theta", theta, False, 0.5, 500, None, None),
            # Limit to Range of NRG G/Ts
            ("g", gamma, True, theta / 1000, theta * 50, None, None),
            ("amp", amp, True, 0.01, 3, None, None),
            ("const", const, True, None, None, None, None),
            ("lin", lin, True, -0.005, 0.005, None, None),
            ("quad", 0, False, None, None, None, None),
            ("occ_lin", occ_lin, False, None, None, None, None),
        )
        return params

    def charge_fit(self, params: Optional[lm.Parameters] = _NOT_SET) -> FitInfo:
        """Do the charge fit, with optional parameters to have more control over max/mins/initial/vary etc"""
        if params is _NOT_SET:
            params = self._last_charge_params  # Even if None
        if not params:
            params = self.make_charge_params()
        
        if self._last_charge_fit is None or self._are_charge_params_new(params):
            self._last_charge_params = params.copy()

            # Fit to the function
            x = self.combined_data.charge.x
            data = self.combined_data.charge.data
            model = lm.models.Model(self.nrg_charge_func) + lm.model.Model(simple_quadratic)
            fit = calculate_fit(
                x=x, data=data, params=params, func=model, method="powell"
            )  # Powell method works best for fitting to interpolated data

            # Cache for quicker return if same params asked again
            self._last_charge_fit = fit
        return self._last_charge_fit

    def plot_charge_fit(self, params: Optional[lm.Parameters] = _NOT_SET, sub_lin=True, show_init=False):
        if params is _NOT_SET:
            params = self._last_charge_params
        elif params != self._last_charge_params:
            logging.warning(f"Plotting charge fit with different parameters to last fit")

        fit = self.charge_fit(params=params)
        x = self.combined_data.charge.x
        data = self.combined_data.charge.data

        fig = default_fig()
        fig.add_trace(go.Scatter(x=x, y=data, mode="markers", name="Data"))
        if show_init:
            fig.add_trace(go.Scatter(x=x, y=fit.eval_init(x=x), mode="lines", name="Initial"))
        fig.add_trace(go.Scatter(x=x, y=fit.eval_fit(x=x), mode="lines", name="Fit"))

        fig.update_layout(title="Charge Transition Fit", yaxis_title='Current /nA')

        if sub_lin:
            m = fit.best_values.lin
            c = fit.best_values.const
            for data in fig.data:
                data.y = data.y - (m * data.x + c)
            fig.update_layout(title=fig.layout.title.text + " sub linear")
        return fig

    def make_conductance_params(self) -> lm.Parameters:
        x = self.combined_data.conductance.x
        data = self.combined_data.conductance.data

        theta = self.theta if theta else 50
        gamma = 0.01
        amp = 1
        const = 0

        # Create lm.parameters with some initial values and limits
        params = lm.Parameters()
        params.add_many(
            ("mid", 0, True, np.nanmin(x), np.nanmax(x), None, None),
            ("theta", theta, False, 0.5, 200, None, None),
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

    def conductance_fit(self, params: Optional[lm.Parameters] = _NOT_SET) -> FitInfo:
        """Do the conductance fit, with optional parameters to have more control over max/mins/initial/vary etc"""
        # First try to get the last run conductance_fit (unless specifying new parameters)
        if params is _NOT_SET:
            params = self._last_conductance_params  # Even if None
        if not params:
            params = self.make_conductance_params()
        
        if self._last_conductance_fit is None or self._are_cond_params_new(params):
            self._last_conductance_params = params.copy()

            # Fit to the function
            x = self.combined_data.conductance.x
            data = self.combined_data.conductance.data
            fit = calculate_fit(
                x=x, data=data, params=params, func=self.nrg_conductance_func, method="powell"
            )  # Powell method works best for fitting to interpolated data

            # Cache for quicker return if same params asked again
            self._last_conductance_fit = fit
        return self._last_conductance_fit

    def plot_conductance_fit(self, params: Optional[lm.Parameters] = _NOT_SET, show_init=False):
        if params is _NOT_SET:
            params = self._last_conductance_params
        elif params != self._last_conductance_params:
            logging.warning(
                f"Plotting conductance fit with different parameters to last fit"
            )

        fit = self.conductance_fit(params=params)
        x = self.combined_data.conductance.x
        data = self.combined_data.conductance.data

        fig = default_fig()
        fig.add_trace(go.Scatter(x=x, y=data, mode="markers", name="Data"))
        if show_init:
            fig.add_trace(go.Scatter(x=x, y=fit.eval_init(x=x), mode="lines", name="Initial"))
        fig.add_trace(go.Scatter(x=x, y=fit.eval_fit(x=x), mode="lines", name="Fit"))

        fig.update_layout(title="Conductance Fit")
        return fig
    
    def _are_charge_params_new(self, params: lm.Parameters):
        if params is None:
            params = self.make_charge_params()
        last_is_none = self._last_charge_params is None
        pars_not_equal = not are_params_equal(params, self._last_charge_params)
        if last_is_none or pars_not_equal:
            return True
        return False
    
    def _are_cond_params_new(self, params: lm.Parameters):
        if params is None:
            params = self.make_conductance_params()
        last_is_none = self._last_conductance_params is None
        pars_not_equal = not are_params_equal(params, self._last_conductance_params)
        if last_is_none or pars_not_equal:
            return True
        return False
    
    
    
@dataclass
class SimultaneousFitResult:
    fit: lm.model.ModelResult
    params_list: list[lm.Parameters]


class SimultaneousFitNRG:
    def __init__(
        self,
        fit_to_nrgs: list[FitToNRG],
        theta_expressions: Optional[list[str]] = None,
        gamma_shared: bool = True,
        linear_shared: bool = True,
        occ_lin_shared: bool = True,
        use_last_fit_as_start: bool = True,
    ):
        """Carry out fitting on multiple datasets simultaneously

        Args:
            theta_expressions: "expr" strings to be passed to lm.Parameters. E.g. [None, "3*theta_0", "5*theta_0"] if the datas are at 100, 300, 500mK respectively
            gamma_shared: If True, all gammas will be forced to be equal.
            use_last_fit_as_start: If True and a previous fit exists, previous fit parameters will be used as starting point
        """
        # Data to fit
        self.fit_to_nrgs = fit_to_nrgs

        # Simultaneous Fitting Parameters
        self.theta_expressions = theta_expressions
        self.gamma_shared = gamma_shared
        self.linear_shared = linear_shared
        self.occ_lin_shared = occ_lin_shared

        # Other fiting parameters
        self.use_last_fit_as_start = use_last_fit_as_start

        # Caching Params
        self._previous_fits: dict[str, lm.model.ModelResult] = {}
        self._previous_params: dict[str, tuple[lm.Parameters]] = {}

    def charge_fit(
        self,
        per_dataset_params: list[lm.Parameters] = _NOT_SET,
        full_params: lm.Parameters = _NOT_SET,
    ) -> SimultaneousFitResult:
        return self.fit(
            which="charge", per_dataset_params=per_dataset_params, full_params=full_params
        )


    def conductance_fit(
        self,
        per_dataset_params: list[lm.Parameters] = _NOT_SET,
        full_params: lm.Parameters = _NOT_SET,
    ) -> SimultaneousFitResult:
        return self.fit(
            which="conductance",
            per_dataset_params=per_dataset_params,
            full_params=full_params,
        )
    
    def fit(
        self,
        which: str = "charge",
        per_dataset_params: list[lm.Parameters] = _NOT_SET,
        full_params: lm.Parameters = _NOT_SET,
    ) -> SimultaneousFitResult:
        """Fit the data simultaneosly"""
        # Return previous fit if possible
        if (
            per_dataset_params is _NOT_SET
            and full_params is _NOT_SET
            and which in self._previous_fits
        ):
            return self._previous_fits[which]

        # Make full_params if necessary
        if full_params in [_NOT_SET, None] and per_dataset_params in [_NOT_SET, None]:
            # Get default params
            full_params = self.combine_params(self.make_per_dataset_params(which=which))
        elif per_dataset_params not in [_NOT_SET, None]:  # Use per dataset params
            full_params = self.combine_params(per_dataset_params)
        else:  # Use full specified params
            full_params = full_params

        # Check if this fit has already been done, in which case return it
        previous_params = self._previous_params.get(which, None)
        if (
            previous_params is not None
            and are_params_equal(full_params, previous_params)
            and which in self._previous_fits
        ):
            return self._previous_fits[which]

        # Collect Data to Fit
        xs, datas = self._collect_datas(which=which)

        # Do fit
        objective = self.get_objective_function(which)
        fit = lm.minimize(
            objective, full_params, method="powell", args=(xs, datas), nan_policy="omit"
        )
        self._previous_fits[which] = SimultaneousFitResult(
            fit=fit, params_list=self._separate_combined_params(fit.params)
        )
        self._previous_params[which] = tuple(full_params)
        return self._previous_fits[which]

    def eval_fit_dataset(
        self, i: int, x: np.ndarray, which: str = "charge"
    ) -> np.ndarray:
        """Evaluate the fit for a single dataset for the given x"""
        fit = self.fit(which=which)
        return self._nrg_func_dataset(fit.fit.params, i, x, which=which)

    def eval_init_dataset(
        self, i: int, x: np.ndarray, which: str = "charge"
    ) -> np.ndarray:
        """Evaluate the fit for a single dataset for the given x"""
        fit = self.fit(which=which)
        initial_values = {k: p.init_value for k, p in fit.fit.params.items()}
        return self._nrg_func_dataset(initial_values, i, x, which=which)

    def plot_fits(
        self,
        which: str = "charge",
        show_init: bool = False,
        waterfall: bool = False,
        waterfall_spacing: float = None,
        sub_linear: bool = None,
        align: bool = False,
    ) -> go.Figure:
        """Plot the full simulaneous fit (i.e. multiple 1D fits)"""
        if sub_linear is None:
            sub_linear = True if which == 'charge' else False
        if sub_linear and not align:
            logging.warning(
                f"Expect bad results when subtracting linear term without aligning first. Linear term determined with center value taken into account"
            )

        xs, datas = self._collect_datas(which=which)

        fig = default_fig()
        previous_data = None
        total_waterfall_spacing = 0
        for i, (x, data, pars) in enumerate(
            zip(xs, datas, self.fit(which=which).params_list)
        ):
            nrg_fit_data = self.eval_fit_dataset(i, x, which=which)
            nrg_init_data = self.eval_init_dataset(i, x, which=which)

            if align:
                x -= pars["mid"].value
            if sub_linear:
                line = lm.models.LinearModel()
                line_pars = line.make_params()
                line_pars["slope"].value = pars["lin"].value
                line_pars["intercept"].value = pars["const"].value

                data -= line.eval(x=x, params=line_pars)
                nrg_fit_data -= line.eval(x=x, params=line_pars)
                nrg_init_data -= line.eval(x=x, params=line_pars)

            if waterfall:
                spacing = 0
                if i != 0:
                    if waterfall_spacing is not None:
                        spacing = waterfall_spacing
                    else:
                        p_interp = interp1d(previous_data.x, previous_data.data)
                        n_interp = interp1d(x, data + total_waterfall_spacing)
                        # n_interp = interp1d(x, data)
                        low_x, high_x = max(min(previous_data.x), min(x)), min(
                            max(previous_data.x), max(x)
                        )
                        _x = np.linspace(low_x, high_x, 1000)

                        # Ensure data does not cross itself
                        spacing = np.nanmax(p_interp(_x) - n_interp(_x))
                        # Add a little more space to make things look nicer
                        spacing += 0.02 * (np.nanmax(data) - np.nanmin(data))

                total_waterfall_spacing += spacing
                data += total_waterfall_spacing
                nrg_fit_data += total_waterfall_spacing
                nrg_init_data += total_waterfall_spacing
                previous_data = Data(x=x, data=data)

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=data,
                    name=f"Data",
                    legendgroup=f"{i}",
                    legendgrouptitle_text=f"{i}",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=nrg_fit_data,
                    name=f"Fit",
                    legendgroup=f"{i}",
                )
            )
            if show_init:
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=nrg_init_data,
                        name=f"Init",
                        legendgroup=f"{i}",
                    )
                )

        return fig

    def get_objective_function(self, which: str):
        """Objectie function to minimize for all data in one go"""
        if which not in ["charge", "conductance"]:
            raise ValueError(
                f'which objective must be in ["charge", "conductance"], got {which} instead'
            )

        def objective(params, xs, datas):
            resids = []
            for i, (row, x) in enumerate(zip(datas, xs)):
                resids.extend(row - self._nrg_func_dataset(params, i, x, which=which))

            # 1D resids required for lm.minimize
            return resids

        return objective

    def _collect_datas(self, which: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Collect the xs and datas from each of the FitToNRG instances"""
        # Note: getattrs just allows for accessing a variable by string name instead e.g. obj.var becomes getattr(obj, "var")
        xs = [
            np.array(getattr(ftn.combined_data, which).x, dtype=np.float32)
            for ftn in self.fit_to_nrgs
        ]
        datas = [
            np.array(getattr(ftn.combined_data, which).data, dtype=np.float32)
            for ftn in self.fit_to_nrgs
        ]
        return xs, datas

    def _separate_combined_params(
        self, combined_params: lm.Parameters
    ) -> list[lm.Parameters]:
        """Separate out the combined simultaneous fitting parameters back into the individual fitting parameters
        Note: will only work up to 10 datasets without modification due to imperfect way of getting max_num and unique
        """
        all_keys = [k for k in combined_params]
        param_nums = [int(re.search("_(\d+)", k).groups()[0]) for k in all_keys]
        max_num = max(param_nums)
        unique = set([re.search("(.+)_\d+", k).groups()[0] for k in all_keys])
        params_list = []
        for i in range(max_num + 1):
            pars = lm.Parameters()
            for k in unique:
                p = combined_params[f"{k}_{i}"]
                vary = False if p.vary is False or p.expr else True
                pars.add(k, value=p.value, min=p.min, max=p.max, vary=vary)
                pars[k].stderr = p.stderr
            params_list.append(pars)
        return params_list

    def _nrg_func_dataset(self, params, i, x, which="charge"):
        """Evaluate NRG func for dataset {i}"""
        which = (
            "i_sense" if which == "charge" else which
        )  # nrg_func expects this name instead for charge

        mid = params[f"mid_{i}"]
        g = params[f"g_{i}"]
        theta = params[f"theta_{i}"]
        amp = params[f"amp_{i}"]
        lin = params[f"lin_{i}"]
        const = params[f"const_{i}"]
        occ_lin = params[f"occ_lin_{i}"]
        data = nrg.nrg_func(x, mid, g, theta, amp, lin, const, occ_lin, data_name=which)
        if which == "i_sense" and params[f'quad_{i}'] != 0:
            data += simple_quadratic(x, params[f'quad_{i}'])
        return data
        
    def combine_params(self, params_list: list[lm.Parameters]) -> lm.Parameters:
        """Combine list of lm.Parameters into one lm.Parameters with unique names"""
        combined_params = lm.Parameters()
        for i, pars in enumerate(params_list):
            for par_name in pars.keys():
                combined_params[f"{par_name}_{i}"] = pars[par_name]
        return combined_params
    
    def _add_expr_to_params(self, params: lm.Parameters) -> lm.Parameters:
        for i in range(len(self.fit_to_nrgs)):
            # Make shared parameters
            if i != 0:
                for k, bool_ in zip(
                    ["g", "lin", "occ_lin"],
                    [self.gamma_shared, self.linear_shared, self.occ_lin_shared],
                ):
                    if bool_:
                        params[f'{k}_{i}'].expr = f"{k}_0"  # Once combined, the parameters have the data index added as "_<index>"

            # Make other changes to parameters
            if self.theta_expressions is not None:
                pars[f"theta_{i}"].expr = self.theta_expressions[i]  # e.g. "5*theta_0" to force this dataset to have theta 5x the first dataset
        return params

    def make_per_dataset_params(self, which: str) -> list[lm.Parameters]:
        """Make a single Parameters object containing fitting parameters for each dataset"""
        params_list = []
        for i, ftn in enumerate(self.fit_to_nrgs):
            # Get initial pars
            pars = self._get_starting_pars(ftn, which)

            # Add to the list of params for each dataset
            params_list.append(pars)
        return params_list
                               
    def make_full_params(self, which: str) -> lm.Parameters:
        params = self.make_per_dataset_params(which=which)
        combined_params = self.combine_params(params)
        full_params = self._add_expr_to_params(combined_params)
        return full_params.copy()
                               

    def _get_starting_pars(self, ftn: FitToNRG, which: str) -> lm.Parameters:
        """Get the starting point for initial parameters for each dataset (i.e. before making changes for simultaneous fitting)"""
        if which not in ["charge", "conductance"]:
            raise ValueError(f'Got {which}, expected "charge" or "conductance"')

        if which == "charge":
            if self.use_last_fit_as_start and ftn._last_charge_fit is not None:
                pars = ftn._last_charge_fit.params.copy()
            else:
                pars = ftn.make_charge_params().copy()
        elif which == "conductance":
            if self.use_last_fit_as_start and ftn._last_conductance_fit is not None:
                pars = ftn._last_conductance_fit.params.copy()
            else:
                pars = ftn.make_conductance_params().copy()
        else:
            raise RuntimeError("Shouldn't be able to reach this")

        return pars

    

class CompareToNRG:
    def __init__(
        self,
        combined_data: CombinedData,
        theta: float,
        gamma: float,
        mid: float,
        charge_amp: float,
        charge_lin: float,
        charge_const: float,
        charge_occ_lin: float,
        charge_quad: float,
        conductance_amp: float,
        conductance_const: float,
    ):
        self.combined_data = combined_data
        # Values that are shared between conductance and charge
        self.theta = theta
        self.gamma = gamma
        self.mid = mid

        # Charge specific fit params
        self.charge_amp = charge_amp
        self.charge_lin = charge_lin
        self.charge_const = charge_const
        self.charge_occ_lin = charge_occ_lin
        self.charge_quad = charge_quad

        # Conductance specific fit params
        self.conductance_amp = conductance_amp
        self.conductance_const = conductance_const

        # Make the charge/conductance params
        self.charge_params = nrg.NRGParams(
            gamma=gamma,
            theta=theta,
            center=mid,
            amp=charge_amp,
            lin=charge_lin,
            const=charge_const,
            lin_occ=charge_occ_lin,
        )
        self.conductance_params = nrg.NRGParams(
            gamma=gamma,
            theta=theta,
            center=mid,
            amp=conductance_amp,
            lin=0,
            const=conductance_const,
            lin_occ=0,
        )

    @classmethod
    def from_FitToNRG(cls, fit_to_nrg: FitToNRG) -> CompareToNRG:
        combined_data = fit_to_nrg.combined_data
        charge_fit = fit_to_nrg.charge_fit()
        conductance_fit = fit_to_nrg.conductance_fit()

        # Raise error if there isn't agreement with g or theta
        for k in ["g", "theta"]:
            if charge_fit.best_values.get(k) != conductance_fit.best_values.get(k):
                raise ValueError(
                    f"{k} does not match for the charge/conductance fits of FitToNRG, please init explicity using CompareToNRG(...)"
                )

        gamma = charge_fit.best_values.g
        theta = charge_fit.best_values.theta
        mid = charge_fit.best_values.mid

        charge_amp = charge_fit.best_values.amp
        charge_lin = charge_fit.best_values.lin
        charge_const = charge_fit.best_values.const
        charge_occ_lin = charge_fit.best_values.occ_lin
        charge_quad = charge_fit.best_values.quad
        
        conductance_amp = conductance_fit.best_values.amp
        conductance_const = conductance_fit.best_values.const
        return cls(
            combined_data=combined_data,
            theta=theta,
            gamma=gamma,
            mid=mid,
            charge_amp=charge_amp,
            charge_lin=charge_lin,
            charge_const=charge_const,
            charge_occ_lin=charge_occ_lin,
            charge_quad=charge_quad,
            conductance_amp=conductance_amp,
            conductance_const=conductance_const,
        )

    def __repr__(self):
        return f"CompareToNRG({repr(self.combined_data)},\n theta={self.theta:.3g}, gamma={self.gamma:.3g}, mid={self.mid:.3g}, charge_amp={self.charge_amp:.3g}, charge_lin={self.charge_lin:.3g}, charge_const={self.charge_const:.3g}, charge_quad={self.charge_quad:.3f}, conductance_amp={self.conductance_amp:.3g}, conductance_const={self.conductance_const:.3g})"

    def plot_full_comparison(self, normalize=True, sub_linear=True):
        figs = []
        figs.append(self.plot_conductance(normalize=normalize))
        figs.append(self.plot_conductance(occupation_x=False, normalize=normalize))
        figs.append(self.plot_charge(occupation_x=False, residual=True))
        figs.append(self.plot_charge(occupation_x=False, sub_linear=sub_linear))
        figs = [fig for fig in figs if fig is not None]
        fig = figures_to_subplots(
            figs,
            cols=4,
            rows=1,
            title=f"Comparison to NRG - {self.params_summary()}",
            shared_data=True,
        )
        return fig

    def params_summary(self):
        return f"Theta={self.theta:.3g}, Gamma={self.gamma:.3g}"

    def get_nrg_data(
        self,
        which: str,
        x: np.ndarray = None,
        occupation_x: bool = False,
        const: float = None,
        amp: float = None,
    ) -> Data:
        """Generate the NRG data for the fit params
        Args:
            x: Optional x-array to generate data for (defaults to self.combined_data.charge.x)
            const: Optionally override the const value used when generating data (defaults to 0)
            amp: Optionally override the amp value used when generating data (defaults to 1)
        """
        which = which.lower()
        x = x if x is not None else self.combined_data.charge.x

        data_name = which
        # Allow alias of i_sense
        if which in ["i_sense", "charge"]:
            data_name = "i_sense"
            const = const if const is not None else self.charge_params.const
            amp = amp if amp is not None else self.charge_params.amp
        elif which in ["entropy", "int_entropy"]:
            data_name = "dndt"  # Entropy data does not exist in NRG calculations
            const = 0
            amp = 1
        else:
            const = const if const is not None else 0
            amp = amp if amp is not None else 1

        data = nrg.NrgUtil(self.charge_params).data_from_params(
            x=x, which_data=data_name, const=const, amp=amp
        )
        if which in ['i_sense', 'charge'] and self.charge_quad is not None and self.charge_quad != 0:
            data.data += simple_quadratic(data.x, self.charge_quad)
        
        if occupation_x:
            x_data = nrg.NrgUtil(self.charge_params).data_from_params(
                x=x,
                which_data="occupation",
            )
            data.x = x_data.data

        # If asking for entropy, convert dndt to entropy
        if which in ["entropy", "int_entropy"]:
            # Warn if calculation will be way off
            if np.any(data.data[[0, -1]] > 0.001 * np.max(data.data)):
                first_percent, last_percent = [
                    data.data[index] / np.max(data.data) * 100 for index in [0, -1]
                ]
                logging.warning(
                    f"May need to evaluate to a larger x-value to obtain an accurate entropy calculation. The first(last) dN/dT values were {first_percent:.2g}({last_percent:.2g})% of the max dN/dT"
                )
            entropy = np.cumsum(data.data)
            entropy = entropy / entropy[-1] * np.log(2)
            data.data = entropy

        return Data(x=data.x, data=data.data)

    def get_nrg_conductance(
        self, x: np.ndarray = None, const: float = None, amp: float = None
    ):
        return self.get_nrg_data(which="conductance", x=x, const=const, amp=amp)

    def get_nrg_occupation(self, x: np.ndarray = None):
        return self.get_nrg_data(which="occupation", x=x)

    def get_nrg_charge_transition(
        self, x: np.ndarray = None, const: float = None, amp: float = None
    ):
        return self.get_nrg_data(which="charge", x=x, const=const, amp=amp)

    def plot_nrg_data(
        self,
        which: str,
        x: np.ndarray = None,
        occupation_x: bool = False,
        const: float = None,
        amp: float = None,
    ):
        which = which.lower()
        data = self.get_nrg_data(
            which=which, x=x, occupation_x=occupation_x, const=const, amp=amp
        )
        fig = default_fig()
        fig.add_trace(go.Scatter(x=data.x, y=data.data, name=which))
        fig.update_layout(title=f"NRG {which.title()}")
        if occupation_x is False:
            fig.update_layout(xaxis_title="Sweepgate/Energy /arb")
        else:
            fig.update_layout(xaxis_title="Occupation")

        if which in ["i_sense", "charge"]:
            fig.update_layout(yaxis_title="Current /nA")
        elif which == "conductance":
            fig.update_layout(yaxis_title="Conductance /2e^2/h")
        elif which == "dndt":
            fig.update_layout(yaxis_title="dN/dT /arb")
        elif which in ["entropy", "int_entropy"]:
            fig.update_layout(yaxis_title="Entropy /kB")
        elif which == "occupation":
            fig.update_layout(yaxis_title="Occupation")

        return fig

    def plot_conductance(
        self, use_const=False, use_amp=True, occupation_x=True, normalize=True
    ):
        const = self.conductance_params.const if use_const else 0
        amp = self.conductance_params.amp if use_amp else 1
        data = copy.copy(self.combined_data.conductance)  # In case I normalize it later
        nrg_data = self.get_nrg_data(
            which="conductance",
            x=data.x,
            occupation_x=occupation_x,
            const=const,
            amp=amp,
        )
        x = nrg_data.x  # Either the same sweepgate x, or occupation
        if normalize:
            data.data = data.data / np.nanmax(data.data)
            nrg_data.data = nrg_data.data / np.max(nrg_data.data)
        fig = default_fig()
        fig.add_trace(go.Scatter(x=x, y=data.data, name="data"))
        fig.add_trace(go.Scatter(x=x, y=nrg_data.data, name="nrg"))
        fig.update_layout(
            title=f"Conductance vs {'Occupation' if occupation_x else 'Sweepgate'}{' (normalized)' if normalize else ''}",
            yaxis_title="Conductance /2e^2/h",
        )
        if occupation_x is False:
            fig.update_layout(xaxis_title="Sweepgate")
        else:
            fig.update_layout(xaxis_title="Occupation")
        return fig

    def plot_charge(self, occupation_x=False, sub_linear=False, residual=False):
        # Get relevant data
        data = copy.copy(self.combined_data.charge)  # In case I change it later
        nrg_data = self.get_nrg_data(
            which="charge",
            x=data.x,
            occupation_x=occupation_x,
        )
        
        # Modify data according to options
        if residual: 
            data.data = data.data-nrg_data.data
            nrg_data.data = nrg_data.data-nrg_data.data  # Just zeros, but still plot to not mess up combining plots
        elif sub_linear:
            line = lm.models.LinearModel()
            pars = line.make_params()
            pars["slope"].value = self.charge_params.lin
            pars["intercept"].value = self.charge_params.const
            data.data = data.data - line.eval(x=data.x, params=pars)
            nrg_data.data = nrg_data.data - line.eval(x=data.x, params=pars)
        x = nrg_data.x  # Either the same sweepgate x, or occupation
        
        # Plot data
        fig = default_fig()
        fig.add_trace(go.Scatter(x=x, y=data.data, name="data"))
        fig.add_trace(go.Scatter(x=x, y=nrg_data.data, name="nrg"))
        if residual:
            title_extra = ' (residuals)'
        elif sub_linear:
            title_extra = ' (sub_linear)'
        else:
            title_extra = ''
        fig.update_layout(
            title=f"Charge vs {'Occupation' if occupation_x else 'Sweepgate'}{title_extra}",
            yaxis_title="Current  /nA",
        )
        if occupation_x is False:
            fig.update_layout(xaxis_title="Sweepgate")
        else:
            fig.update_layout(xaxis_title="Occupation")
        return fig