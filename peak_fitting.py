"""
2023-03-28 -- Tim: I don't think it's worth moving this into dat_analysis as it is... I think to make it more
general, this should be done as subclass of GeneralFitter. Some things from here are probably worth using as inspiration
 for a future function.
 Maybe there should be a subclass of GeneralFitter made for repeat measurements (i.e. for many 1D fits to 2D data).
"""
from lmfit.models import LorentzianModel, ConstantModel
from dat_analysis.analysis_tools.general_fitting import FitInfo
from new_util import Data, DatHDF
import plotly.graph_objects as go
import numpy as np
import lmfit as lm
import logging 

from dat_analysis.core_util import get_data_index
from new_util import are_params_equal
from dat_analysis.plotting.plotly.util import default_fig, heatmap 

_NOT_SET = object()


class PeakFitting:
    """
    For working with and plotting data with a peak (e.g. conductance measurements)
    """

    def __init__(
        self,
        data: Data,
        peak_loc: float = None,
        peak_width: float = None,
    ):
        self.data = data
        if self.data.data.ndim != 2:
            logging.warning(f'Peak Fitting expects 2D data, attempting conversion')
            if self.data.data.ndim != 1:
                raise ValueError(f"Peak fitting got data with shape {self.data.data.shape}. Expected 2D")
            self.data = self.data.copy()
            self.data.data = np.atleast_2d(self.data.data)
            self.data.y = np.arange(self.data.data.shape[0])

        # Processing variables
        self.peak_loc = (
            peak_loc
            if peak_loc
            else self.data.x[
                np.where(np.abs(self.data.mean().data) == np.nanmax(np.abs(self.data.mean().data)))[-1]
            ][0]
        )
        self.peak_width = (
            peak_width
            if peak_width
            else (np.nanmax(self.data.x) - np.nanmin(self.data.x)) / 5
        )

        # Caching variables
        self._last_params = None
        self._fits = None

    @classmethod
    def from_dat(
        cls, dat: DatHDF, data_name="conductance_2d", peak_loc=None, peak_width=None
    ):
        x = dat.Data.x
        y = dat.Data.y
        data = dat.Data.get_data(data_name)
        inst = cls(
            data=Data(x=x, y=y, data=data),
            peak_loc=peak_loc,
            peak_width=peak_width,
        )
        return inst

    def plot_2d(self) -> go.Figure:
        """
        Plot the 2D data as recorded (i.e. with repeats)
        """
        # Make figure
        fig = go.Figure()
        fig.add_trace(heatmap(x=self.data.x, y=self.data.y, data=self.data.data))
        fig.update_layout(title=f"Peak Fitting Data 2D")
        return fig

    def plot_fits(
        self,
        slice=np.s_[:1],
        plot_init: bool = False,
        waterfall_spacing=None,
    ):
        """
        Plot the fits to peaks
        Args:
            slice: To select which fits to plot (e.g. np.s_[::10] for every 10th, or np.s_[[7,12]] to show 7th and 12th)
            plot_init: Whether to add initial fit to plots
        """
        if waterfall_spacing is None:
            waterfall_spacing = np.nanmax(self.data.data) * 0.1

        fits = np.array(self.fit())

        x = self.data.x
        data = self.data.data
        fit_indexes = get_data_index(
            x, [self.peak_loc - self.peak_width, self.peak_loc + self.peak_width]
        )
        x_fit = x[fit_indexes[0] : fit_indexes[1]]

        fig = default_fig()
        for i, (d, fit, y_val) in enumerate(
            zip(data[slice], fits[slice], self.data.y[slice])
        ):
            y_shift = i * waterfall_spacing
            n = f"{y_val:.3g}"
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=d + y_shift,
                    name=n,
                    legendgroup=n,
                    mode="markers+lines",
                    line_width=1,
                    marker=dict(symbol="cross", size=5),
                )
            )
            for x_, color, name in zip(
                [x, x_fit], ["dark orange", "orange"], [n + " fit", ""]
            ):
                fig.add_trace(
                    go.Scatter(
                        x=x_,
                        y=fit.eval(x=x_) + y_shift,
                        name=name,
                        legendgroup=n,
                        showlegend=False,
                        mode="lines",
                        line_color=color,
                    )
                )
            if plot_init:
                fig.add_trace(
                    go.Scatter(
                        x=fit.userkws["x"],
                        y=fit.init_fit + y_shift,
                        name=n + " init",
                        legendgroup=n,
                        showlegend=False,
                        mode="lines",
                        line_color="red",
                    )
                )
        fig.update_layout(title="Peak Fits")
        return fig

    def plot_averaged(
        self,
    ) -> go.Figure:
        """Plot averaged conductance data"""
        avg_data = self.data.mean(centers=self.get_centers())
        return avg_data.plot()
    
    def plot_aligned(self) -> go.Figure:
        return self.data.center(centers=self.get_centers()).plot()

    def _default_model(self):
        """Note: Can't store a model as it can't be pickled (breaks Dash App stuff)"""
        return lm.models.LorentzianModel() + lm.models.ConstantModel()

    def make_params(self, model=None):
        if model is None:
            model = self._default_model()
        params = model.make_params()

        sigma_guess = self.peak_width / 10
        dmax = np.nanmax(np.abs(self.data.data)) * np.sign(np.nanmean(self.data.data))
        amp_guess = dmax * sigma_guess / 0.3183
        params["center"].min = self.peak_loc - self.peak_width
        params["center"].max = self.peak_loc + self.peak_width
        params["center"].value = self.peak_loc
        # params["amplitude"].value = (
        #     10 * np.nanmax(np.abs(self.data.data)) * np.sign(np.nanmean(self.data.data))
        # )
        params["amplitude"].value = amp_guess
        # params["amplitude"].min = 10 * np.nanmin(data)
        # params["amplitude"].max = np.nanmax(data) - np.nanmean(data)
        params["sigma"].value = sigma_guess
        params["sigma"].max = sigma_guess * 10
        params["sigma"].min = sigma_guess * 0.01
        params["c"].vary = True
        return params

    def fit(
        self,
        params=_NOT_SET,
        model=None,
    ) -> list[FitInfo]:
        """Find the nearest peak to peak_loc in each repeat"""
        if params is _NOT_SET:
            params = self._last_params  # Even if None
        if model is None:
            model = self._default_model()
        if not params:
            params = self.make_params(model=model)
            
        if self._fits is None or self._are_params_new(params):
            # Copy so that they aren't changed by fitting (i.e. for comparison if called again)
            self._last_params = params.copy()
            x = self.data.x
            data = self.data.data

            indexes = get_data_index(
                x, [self.peak_loc - self.peak_width, self.peak_loc + self.peak_width]
            )
            x = x[indexes[0] : indexes[1]]
            data = data[:, indexes[0] : indexes[1]]

            fits = []
            for d in data:
                # fit = calculate_fit(x=x, data=d, params=params, func=gauss)
                fit = model.fit(
                    np.array(d, dtype=np.float32),
                    x=np.array(x, dtype=np.float32),
                    params=params,
                )
                fits.append(fit)
            self._fits = fits
        return self._fits

    def get_centers(self) -> list[float]:
        fits = self.fit()
        return [f.params["center"].value for f in fits]

    def _are_params_new(self, params: lm.Parameters):
        if params is None:
            params = self.make_params()
        last_is_none = self._last_params is None
        pars_not_equal = not are_params_equal(params, self._last_params)
        if last_is_none or pars_not_equal:
            return True
        return False