"""
2023-03-28 -- Tim: I don't think it's worth moving this into dat_analysis as it is... I think to make it more
general, the centering/aligning should be handled separately to the Conductance fitting. I.e. ConductanceFitting
should be done as subclass of GeneralFitter. Some things from here are probably worth using as inspiration for a
future function

Class for dealing with 2D conductance data (i.e. aligning based on peaks to find average)
"""
from __future__ import annotations
import numpy as np
import plotly.graph_objects as go
from typing import TYPE_CHECKING
from lmfit.models import LinearModel, LorentzianModel

from dat_analysis.analysis_tools.general_fitting import FitInfo
import dat_analysis.useful_functions as U
from dat_analysis.plotting.plotly.util import heatmap, figures_to_subplots, default_fig, error_fill

from new_util import Data, PlottingInfo


if TYPE_CHECKING:
    from dat_analysis.dat.dat_hdf import DatHDF


class Conductance:
    """
    For working with and plotting conductance measurements where the y-axis is repeats (e.g. with small AC lock-in signal to measure conductance very close to zero bias)
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        data: np.ndarray,
        plotting_info: PlottingInfo = None,
        peak_loc: float = None,
        peak_width: float = None,
    ):
        self.x = x
        self.y = y
        self.data = data
        self.data_ = Data(data=data, x=x, y=y)

        # Processing variables
        self.peak_loc = peak_loc
        self.peak_width = peak_width

        self.plotting_info = plotting_info

        # Caching variables
        self._peak_fits = None

    @classmethod
    def from_dat(
        cls, dat: DatHDF, data_name="conductance_2d", peak_loc=None, peak_width=None
    ):
        x = dat.Data.x
        y = dat.Data.y
        data = dat.Data.get_data(data_name)
        plotting_info = PlottingInfo.from_dat(dat)
        inst = cls(
            x=x,
            y=y,
            data=data,
            plotting_info=plotting_info,
            peak_loc=peak_loc,
            peak_width=peak_width,
        )
        return inst

    def plot_2d(self, x=None, aligned=True) -> go.Figure:
        """Plot the 2D data as recorded (i.e. with repeats)

        Optionally after aligning based on peak fits
        """
        if aligned:
            data = self.get_aligned_data()
        else:
            data = Data(data=self.data, x=self.x, y=self.y)

        # Use specified x-axis?
        if x is not None:
            if data.data.shape[1] == x.shape[0]:
                data.x = x
            else:
                data.x = U.get_matching_x(x, data.data)

        # Make figure
        fig = go.Figure()
        fig.add_trace(heatmap(x=data.x, y=data.y, data=data.data))
        if self.plotting_info:
            fig.update_layout(
                title='Conductance 2D',
                xaxis_title=self.plotting_info.x_label,
                yaxis_title=self.plotting_info.y_label,
            )
            if self.plotting_info.datnum:
                fig.update_layout(title=f"Dat{self.plotting_info.datnum}: {fig.layout.title.text}")
        else:
            fig.update_layout(title=f"Conductance 2D")
        fig.update_layout(coloraxis_colorbar_title="2e^2/h")
        return fig

    def plot_fits(
        self,
        max_num: int=10,
        plot_init: bool = False,
    ):
        """Plot the fit to peaks
        Args:
            max_num: Max number of peak fits to plot (i.e. can't really look at 1000 fits in one plot)
            plot_init: Whether to add initial fit to plots
        """
        fits = self.find_peaks()

        x = self.x
        data = self.data
        indexes = U.get_data_index(
            x, [self.peak_loc - self.peak_width, self.peak_loc + self.peak_width]
        )
        x = x[indexes[0] : indexes[1]]
        data = data[:, indexes[0] : indexes[1]]

        waterfall_spacing = np.nanmean(data)*0.1
        
        fig = go.Figure()
        for i, (d, fit, y_val) in enumerate(zip(data, fits, self.y)):
            if max_num is not None and i >= max_num:
                break
            y_shift = i*waterfall_spacing
            n = f"{y_val:.3g}"
            fig.add_trace(go.Scatter(x=x, y=d+y_shift, name=n, legendgroup=n, mode="markers+lines", line_width=1, marker=dict(symbol='cross', size=5)))
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=fit.eval_fit(x=x)+y_shift,
                    name=n + " fit",
                    legendgroup=n,
                    showlegend=False,
                    mode="lines",
                    line_color="green",
                )
            )
            if plot_init:
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=fit.eval_init(x=x)+y_shift,
                        name=n + " init",
                        legendgroup=n,
                        showlegend=False,
                        mode="lines",
                        line_color="red",
                    )
                )
        if max_num is not None:
            fig.update_layout(title=f'First {max_num} peak fits')
        else:
            fig.update_layout(title='Peak Fits')
        return fig

    def plot_averaged(
        self,
    ) -> go.Figure:
        """Plot averaged conductance data"""
        avg_data = self.get_averaged_data()

        fig = default_fig()
        fig.add_trace(
            go.Scatter(
                x=avg_data.x,
                y=avg_data.data,
                mode="lines+markers",
                error_y=avg_data.yerr,
            )
        )
        if len(avg_data.x) > 300:  # Too many error bars, replace with fill
            fig.update_traces(error_y=None)
            fig.add_trace(
                error_fill(x=avg_data.x, data=avg_data.data, error=avg_data.yerr)
            )
        return fig

    def find_peaks(
        self,
    ) -> list[FitInfo]:
        """Find the nearest peak to peak_loc in each repeat"""
        if self._peak_fits is None:
            if self.peak_loc is None:
                # self.peak_loc = np.nanmean(self.x)  # Guess middle of scan
                # Guess highest peak of data
                self.peak_loc = self.x[np.where(self.data == self.data.max())[-1]][0]
            if self.peak_width is None:
                self.peak_width = (np.nanmax(self.x) - np.nanmin(self.x)) / 5

            x = self.x
            data = self.data
            indexes = U.get_data_index(
                x, [self.peak_loc - self.peak_width, self.peak_loc + self.peak_width]
            )
            x = x[indexes[0] : indexes[1]]
            data = data[:, indexes[0] : indexes[1]]

            model = LorentzianModel() + LinearModel()
            params = model.make_params()
            params["center"].min = np.nanmin(x).astype(np.float32)
            params["center"].max = np.nanmax(x).astype(np.float32)
            params["center"].value = self.peak_loc
            params["amplitude"].value = 10 * np.nanmax(data)
            params["amplitude"].min = 0
            # params["amplitude"].max = np.nanmax(data) - np.nanmean(data)
            params["sigma"].value = self.peak_width / 10
            params["sigma"].max = self.peak_width * 5
            params["sigma"].min = self.peak_width * 0.01
            params["slope"].vary = False
            params["intercept"].vary = True
            params["slope"].value = 0

            fits = []
            for d in data:
                # fit = calculate_fit(x=x, data=d, params=params, func=gauss)
                fit = model.fit(
                    np.array(d, dtype=np.float32),
                    x=np.array(x, dtype=np.float32),
                    params=params,
                )
                fit = FitInfo.from_fit(fit)
                fits.append(fit)
            self._peak_fits = fits
        return self._peak_fits

    def get_aligned_data(
        self,
    ) -> Data:
        """Return data after aligning based on peak locations found with self.find_peaks"""
        fits = self.find_peaks()
        centers = [fit.best_values.center for fit in fits]

        x = self.x
        data = self.data
        data_centered, x_centered = U.center_data(
            x=x, data=data, centers=centers, return_x=True
        )
        return Data(x=x_centered, data=data_centered, y=self.y)

    def get_averaged_data(
        self,
    ) -> Data:
        """Return averaged aligned data"""
        fits = self.find_peaks()
        centers = [fit.best_values.center for fit in fits]
        x = self.x
        data = self.data
        data_avg, x_avg, std = U.mean_data(
            x=x, data=data, centers=centers, return_x=True, return_std=True
        )
        return Data(x=x_avg, data=data_avg, yerr=std)