"""
2023-03-28 -- Moved all generally useful functions to dat_analysis (and imported here for backwards
compatability). The rest of the functions left in here are not general enough or require significant re-writing to be
included in dat_analysis.
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly.colors
import plotly.colors as pc
import numpy as np
import pandas as pd
from itertools import product
from dataclasses import dataclass
from typing import Callable, Iterable
import numbers
import os
import lmfit as lm
# from progressbar import progressbar  # Use tqdm instead
from concurrent.futures import ThreadPoolExecutor
import json
from dataclasses import dataclass
from functools import cache
from IPython.display import display, Image
import base64
import logging
import socket
from typing import Any
from tqdm.auto import tqdm

from dat_analysis.characters import PM 

import dat_analysis
from dat_analysis import useful_functions as U
from dat_analysis.dat.dat_hdf import DatHDF


from dat_analysis.analysis_tools.general import ColumnDescriptor, dats_to_df, get_unique_df_vals, summarize_df
from dat_analysis.plotting.plotly.util import default_config, default_layout, default_fig, apply_default_layout, heatmap, error_fill, figures_to_subplots, get_colors  # Initially made here, in future these should be imported directly from dat_analysis.plotting.plotly.util

from new_util import make_animated, Data, PlottingInfo, fig_waterfall, clean_ct, are_params_equal
from dat_analysis.dash.util import get_unused_port as find_open_port
from dat_analysis.useful_functions import mm


##### Transition class import #####
from dat_analysis.analysis_tools.transition import (
    CenteredAveragingProcess,
    TransitionFitProcess,
    get_param_estimates,
    calculate_fit,
    i_sense,
)

##### EXTRA RANDOM LIBRARY IMPORT #####
from scipy.signal import savgol_filter


pio.renderers.default = 'plotly_mimetype+notebook+pdf'  # Allows working in jupyter lab, jupyter notebook, and export to pdf

fig_dir = 'figures/'
os.makedirs(fig_dir, exist_ok=True)


def get_dat(datnum, raw=False, overwrite=False):
    return dat_analysis.get_dat(datnum, host_name='qdev-xld', user_name='Tim', experiment_name='202211_KondoEntropy', raw=raw, overwrite=overwrite)


def get_dats(datnums):
    return [get_dat(num) for num in datnums]


def get_dats_that_meet_conditions(
    datnums: Iterable[int],
    dat_checking_func: Callable[[DatHDF], bool] = None,
    verbose=True,
) -> tuple[DatHDF]:
    """Go through list of datnums and collect the dats that meet the criteria of dat_checking_func (if provided)"""
    def log(message, level=logging.INFO):
        if verbose:
            logging.log(level, message)

    good_dats = []
    for num in datnums:
        try:
            dat = get_dat(num)
            if dat_checking_func is None or dat_checking_func(dat):
                good_dats.append(dat)
            else:
                log(f"Dat{num} did not pass dat_checking_func")
        except Exception as e:
            log(f"Dat{num} failed and raised {e}", level=logging.WARNING)
    return tuple(good_dats)


class Diamonds:
    """For plotting conductance data where the x-axis is DC bias
    Mostly, add the function of smoothing and differentiating the data along with some pre-built plotting functions
    """
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        data: np.ndarray,
        resample_num: int = 300,
        smooth_num_x: int = None,
        smooth_poly_order_x: int = 3,
        smooth_num_y: int = None,
        smooth_poly_order_y: int = 3,
        plotting_info: PlottingInfo = None,
    ):
        self.x = x
        self.y = y
        self.data = data
        self.resample_num = resample_num
        self.smooth_num_x = smooth_num_x
        self.smooth_poly_order_x = smooth_poly_order_x
        self.smooth_num_y = smooth_num_y
        self.smooth_poly_order_y = smooth_poly_order_y
        self.plotting_info = plotting_info

    @classmethod
    def from_dat(cls, 
                 dat: DatHDF, 
                 data_name="dotcurrent_2d",
                 resample_num: int = 300,
                 smooth_num: int = 3,):
        x = dat.Data.x
        y = dat.Data.y
        data = dat.Data.get_data(data_name)
        plotting_info = PlottingInfo.from_dat(dat)
        inst = cls(
            x=x, 
            y=y, 
            data=data, 
            resample_num=resample_num,
            smooth_num=smooth_num,
            plotting_info=plotting_info)
        return inst

    def plot_conductance(self, x=None, smoothed=False):
        # Use smoothed data?
        if smoothed:
            data = self.resample_and_smooth_data()
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
                title=f"Dat{self.plotting_info.datnum}: Conductance 2D",
                xaxis_title=self.plotting_info.x_label,
                yaxis_title=self.plotting_info.y_label,
            )
        else:
            fig.update_layout(title=f"Conductance 2D")
        return fig

    def plot_differentiated_conductance(
        self,
    ):
        data = self.differentiate_data()

        # Make figure
        fig = go.Figure()
        fig.add_trace(heatmap(x=data.x, y=data.y, data=data.data))
        if self.plotting_info:
            fig.update_layout(
                title=f"Dat{self.plotting_info.datnum}: Differential Conductance 2D",
                xaxis_title=self.plotting_info.x_label,
                yaxis_title=self.plotting_info.y_label,
            )
        else:
            fig.update_layout(title=f"Differential Conductance 2D")
        fig.update_traces(showscale=False)
        return fig

    def resample_and_smooth_data(
        self,
    ) -> Data:
        """Resample down to X datapoints, then smooth X and Y datapoints"""
        data = U.decimate(self.data, numpnts=self.resample_num)
        for num, order, axis_data, axis_dim in zip(
            [self.smooth_num_x, self.smooth_num_y],
            [self.smooth_poly_order_x, self.smooth_poly_order_y],
            [self.x, self.y],
            [1, 0],
        ):
            if num and num > order:
                data = savgol_filter(
                    data, window_length=num, polyorder=order, axis=axis_dim
                )
        y = U.get_matching_x(self.y, shape_to_match=data.shape[0])
        x = U.get_matching_x(self.x, data)
        return Data(data=data, x=x, y=y)
    

    def differentiate_data(
        self,
    ):
        smoothed_data = self.resample_and_smooth_data()
        diff = np.diff(smoothed_data.data, axis=1)
        x = U.get_matching_x(smoothed_data.x, diff)
        return Data(data=diff, x=x, y=smoothed_data.y)


####################
# Transition Class #
####################

class Transition:
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        data: np.ndarray,
        theta_outlier_std=1,
        center_outlier_std=4,
        plotting_info: PlottingInfo = None,
    ):
        self.x = x
        self.y = y
        self.data = data
        self.theta_outlier_std = theta_outlier_std
        self.center_outlier_std = center_outlier_std
        self.plotting_info = plotting_info

        # self._centering = None
        self._centering_fits = None
        self._avg_data = None
        self._last_provided_params = None

    @classmethod
    def from_dat(
        cls,
        dat: DatHDF,
        data_name="standard/i_sense",
        theta_outlier_std=1,
        center_outlier_std=4,
    ):

        x = dat.Data.x
        y = dat.Data.y
        data = dat.Data.get_data(data_name)

        if data.ndim == 1:
            data = np.atleast_2d(data)
            y = np.array([0])

        plotting_info = PlottingInfo.from_dat(dat)
        inst = cls(
            x=x,
            y=y,
            data=data,
            plotting_info=plotting_info,
            theta_outlier_std=theta_outlier_std,
            center_outlier_std=center_outlier_std,
        )
        return inst

    def make_params(self, x:np.ndarray = None, data: np.ndarray = None):
        if x is None:
            x = self.x
        if data is None:
            data = self.data[0]
        params = get_param_estimates(self.x, self.data[0])
        # Note: Used to force lin positive, but with virtual gates that doesn't make sense
        params['lin'].min = -np.inf
        return params
    
    def _check_params_match_or_none(self, params):
        """
        Check if params are none or match previously provided. If so, return True
        If not, return False and update self._last_provided_params
        
        Using this to decide whether cachces need to be cleared (e.g. delete old self._centering_fits if using new params)
        """
        if params is None:
            # No need to check anything else
            return True
        if self._last_provided_params is None:
            # First provided params, so store and return OK
            self._last_provided_params = params
            return True
        if are_params_equal(params, self._last_provided_params):
            return True
        self._last_provided_params = params
        return False
    
    def get_centering(
        self, params=None
    ):
        """
            Run fits to determine centers
        """
        if not self._check_params_match_or_none(params):
            self._centering_fits = None
            self._avg_data = None
            
        if self._centering_fits is None:
            if params is None:
                 params = self.make_params()
            fits = [calculate_fit(self.x, d, params=params, func=i_sense) for d in self.data]
            self._centering_fits = fits
        return self._centering_fits
    
#         if self._centering is None:
#             pars = get_param_estimates(self.x, self.data[0])
#             pars["mid"].value = 50
#             centering = CenteredAveragingProcess()
#             centering.set_inputs(x=self.x, datas=self.data, initial_params=pars)
#             centering.process()
#             self._centering = centering

#         return self._centering

    def get_avg_data(
        self,
    ):
        """
        return avergae data, if theta_outlier_std is specified as None then include all data
        """
        if self._avg_data is None:
            good_thetas_index = np.arange(0, self.data.shape[0], 1)
            if self.theta_outlier_std is not None:
                good_thetas_index = self._get_theta_inliers_index()

            # remove centres outside of data range
            good_centers_index = self._get_centers_inside_xdata_index() 
            if self.center_outlier_std is not None:
                good_centers_index = (
                    self._get_centers_inliers_index()
                )  # remove centres outtside of datarange and self.center_outlier_std deviations from average


            good_index = np.array(
                list(set(good_thetas_index).intersection(good_centers_index))
            )

            center_fits = self.get_centering()
            if good_index.size >= 1 and center_fits:  # if no good rows from data
                x = self.x
                data = self.data[good_index]
                centers = np.array([fit.best_values.mid for fit in center_fits])[
                        good_thetas_index
                    ]
                data_avg, x_avg = U.mean_data(
                        x=x, data=data, centers=centers, return_x=True
                    )

                self._avg_data = x_avg, data_avg
            else:
                data_avg, x_avg = np.nanmean(self.data, axis=0), self.x
                logging.warning('Data not aligned before averaging')
                self._avg_data = x_avg, data_avg
        return self._avg_data


    def get_avg_fit(
        self, params = None
    ):
        """
        return fit from `get_avg_data(self, )`
        """
        x, data = self.get_avg_data()
        if params is None:
            params = self.make_params(x=x, data=data)
        fit = calculate_fit(x, data, params=params, func=i_sense)
        return fit


    @property
    def theta(self):
        try:
            fit = self.get_avg_fit()
            theta_param = fit["params"]["theta"]
            return theta_param.value
        except:
            return np.nan

    @property
    def theta_std(self):
        try:
            fit = self.get_avg_fit()
            theta_param = fit["params"]["theta"]
            return theta_param.stderr
        except:
            return np.nan

    def get_thetas_from_centering(
        self,
    ):
        """
        return array of theta values from centering output
        """
        center_fits = self.get_centering()
        thetas = np.array([fit.best_values.theta for fit in center_fits])
        return thetas

    def get_thetas_std_from_centering(
        self,
    ):
        """
        return array of theta values from centering output
        """
        center_fits = self.get_centering()
        thetas_std = np.array([fit.params['theta'].stderr for fit in center_fits])
        return thetas_std

    def _get_centers_inside_xdata_index(
        self,
    ):
        """
        return indexs of data where centre of transition is inside x-range
        """
        center_fits = self.get_centering()
        centers = np.array([fit.best_values.mid for fit in center_fits])
        
        x = self.x
        minx, maxx = np.min(x), np.max(x)

        good_centers_index = []
        for i, center in enumerate(centers):
            if center >= minx and center <= maxx:
                good_centers_index.append(i)

        if len(good_centers_index) / len(centers) < 0.1:
            logging.warning(
                f"{100 - len(good_centers_index)/len(centers)*100:.0f}% data centre fit outside of x range :: bad center"
            )

        return np.array(good_centers_index)

    def _get_centers_inliers_index(
        self,
    ):
        """
        return index of centres which are within self.center_outlier_std of the average center
        from centers inside the x-range of available data
        """

        center_fits = self.get_centering()
        centers = np.array([fit.best_values.mid for fit in center_fits])
        
        x = self.x
        centers_inside_data_index = self._get_centers_inside_xdata_index()

        if centers_inside_data_index.size < 1:
            return []
        else:
            centers_inside_data = centers[centers_inside_data_index]
            avg_center = np.average(centers_inside_data)
            std_center = np.std(centers_inside_data)

            good_centre_index = []

            for index, center in enumerate(centers):
                if (
                    abs(center - avg_center) - self.center_outlier_std * std_center <= 0
                    and index in centers_inside_data_index
                ):
                    good_centre_index.append(index)

            return np.array(good_centre_index)

    def _get_theta_outliers_index(
        self,
    ):
        """
        return array of indexes of thetas outside of self.theta_outlier_std
        """
        thetas = self.get_thetas_from_centering()
        avg_theta = np.average(thetas)
        std_theta = np.std(thetas)

        # Figure out outliers
        indexs = np.where(
            abs(thetas - avg_theta) - self.theta_outlier_std * std_theta > 0
        )[0]
        if len(indexs) / len(thetas) > 0.1:
            logging.warning(
                f"{len(indexs)/len(thetas)*100:.0f}% data thrown out :: bad theta"
            )

        return indexs

    def _get_theta_inliers_index(
        self,
    ):
        """
        return array of indexes of thetas inside of self.theta_outlier_std
        """
        thetas = self.get_thetas_from_centering()
        avg_theta = np.average(thetas)
        std_theta = np.std(thetas)

        # Figure out inliers
        return np.where(
            abs(thetas - avg_theta) - self.theta_outlier_std * std_theta <= 0
        )[0]

    def plot_2d_raw(
        self,
    ):
        """
        return raw data in heatmap
        """
        fig = default_fig()
        fig.add_trace(heatmap(self.x, self.y, self.data))  # Tim's heatmap
        if self.plotting_info:
            fig.update_layout(
                title=f"Dat{self.plotting_info.datnum}: Conductance 2D",
                xaxis_title=self.plotting_info.x_label,
                yaxis_title="Repeats",
            )
        else:
            fig.update_layout(title=f"2D Data ")
        return fig

    def plot_2d_with_centers(
        self,
    ):
        """
        return heatmap with marked centres of transition
        """
        fig = default_fig()

        centers = np.array([fit.best_values.mid for fit in self.get_centering()])

        good_centers_index = np.arange(0, self.data.shape[0], 1)
        if self.theta_outlier_std is not None:
            good_centers_index = (
                self._get_centers_inliers_index()
            )  # remove centres outtside of datarange and self.center_outlier_std deviations from average
            fig_title = f"Dat{self.plotting_info.datnum}: Conductance 2D, centers > {self.center_outlier_std} std, thrown out "
        else:
            good_centers_index = (
                self._get_centers_inside_xdata_index()
            )  # remove centres outside of data range
            fig_title = f"Dat{self.plotting_info.datnum}: Conductance 2D, centers outside of data x-range removed"

        bad_thetas_index = self._get_theta_outliers_index()

        all_index = np.arange(0, len(centers), 1)
        bool_good_centers = np.isin(all_index, good_centers_index)
        bool_bad_centers = np.logical_not(bool_good_centers)

        fig.add_trace(heatmap(self.x, self.y, self.data))  # Tim's heatmap
        fig.add_trace(
            go.Scatter(
                x=centers[bool_good_centers],
                y=self.y[bool_good_centers],
                showlegend=False,
                mode="markers",
                marker=dict(symbol="cross", color="white"),
                name="good data",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=centers[bool_bad_centers],
                y=self.y[bool_bad_centers],
                showlegend=False,
                mode="markers",
                marker=dict(symbol="cross", color="red"),
                name="bad center",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=centers[bad_thetas_index],
                y=self.y[bad_thetas_index],
                showlegend=False,
                mode="markers",
                marker=dict(symbol="cross", color="black"),
                name="bad theta",
            )
        )

        if self.plotting_info:
            fig.update_layout(
                title=fig_title,
                xaxis_title=self.plotting_info.x_label,
                yaxis_title="Repeats",
            )
        else:
            fig.update_layout(title=fig_title)
        return fig

    def plot_outlier_thetas(
        self,
    ):
        """
        return a figure of the theta fits from each fit row where thetas within self.theta_outlier_std std's are included (in green)
        """

        fig = default_fig()

        thetas = self.get_thetas_from_centering()
        thetas_std = self.get_thetas_std_from_centering()
        theta_avg = np.average(thetas)
        theta_std = np.std(thetas)
        theta_inlier_index = np.arange(0, len(thetas), 1)

        if self.theta_outlier_std is not None:

            # add outliers to figure
            theta_outlier_index = self._get_theta_outliers_index()
            x, y, error = (
                self.y[theta_outlier_index],
                thetas[theta_outlier_index],
                thetas_std[theta_outlier_index],
            )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="markers",
                    name=f"thetas > {self.theta_outlier_std}std",
                    line_color="red",
                    error_y=dict(type="data", array=error, visible=True),
                )
            )

            fig.add_hline(
                y=theta_avg - theta_std * self.theta_outlier_std,
                opacity=1,
                line_width=3,
                line_color="red",
                line_dash="dash",
            )
            fig.add_hline(
                y=theta_avg + theta_std * self.theta_outlier_std,
                opacity=1,
                line_width=3,
                line_color="red",
                line_dash="dash",
            )

            theta_inlier_index = self._get_theta_inliers_index()

        # add inliers to figure
        x, y, error = (
            self.y[theta_inlier_index],
            thetas[theta_inlier_index],
            thetas_std[theta_inlier_index],
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                name=f"thetas < {self.theta_outlier_std}std",
                line_color="green",
                error_y=dict(type="data", array=error, visible=True),
            )
        )

        fig.add_hline(
            y=theta_avg, opacity=1, line_width=3, line_color="green", line_dash="dash"
        )

        if self.plotting_info:
            fig.update_layout(
                title="2D Data (resampled) with centers",
                xaxis_title=self.plotting_info.y_label,
                yaxis_title="Thetas",
            )
        else:
            fig.update_layout(title=f"2D Data (resampled) with centers")
        return fig

    def plot_transition_outliers(
        self,
    ):
        """
        return a figure of the transitions of the outlier thetas
        """

        fig = default_fig()

        if self.theta_outlier_std is not None:

            bad_thetas_index = self._get_theta_outliers_index()
            x, y, data = self.x, self.y[bad_thetas_index], self.data[bad_thetas_index]

            if len(bad_thetas_index) > 0:

                fig.add_trace(heatmap(x, y, data))  # Tim's heatmap
                fig = fig_waterfall(fig, waterfall_state=True)
                fig.update_traces(showlegend=False)

            if self.plotting_info:
                fig.update_layout(
                    title=f"BAD Fits with theta > {self.theta_outlier_std} standard deviation",
                    xaxis_title=self.plotting_info.x_label,
                    yaxis_title="Current /nA",
                )
            else:
                fig.update_layout(
                    title=f"BAD Fits with theta > {self.theta_outlier_std} standard deviation"
                )
        return fig

    def plot_avg_data(
        self,
    ):
        """
        return a figure of the averaged data where outlier thetas have been excluded from the averaging
        """
        fig = default_fig()

        try:
            x_avg, data_avg = self.get_avg_data()

            fit = self.get_avg_fit()

            theta, theta_std = self.theta, self.theta_std

            fig.add_trace(
                go.Scatter(x=x_avg, y=data_avg, mode="lines", name="avg data")
            )
            fig.add_trace(
                go.Scatter(x=x_avg, y=fit.eval_fit(x_avg), mode="lines", name="fit")
            )

            if self.plotting_info:
                fig.update_layout(
                    title=f"Averaged Data from Good Fits theta ={theta:.2f} {PM} {theta_std:.2f}",
                    xaxis_title=self.plotting_info.x_label,
                    yaxis_title="Current /nA",
                )
            else:
                fig.update_layout(title=f"Averaged Data from Good Fits")
            return fig
        except:
            fig.add_annotation(
                xref="x domain",
                yref="y domain",
                x=0.5,
                y=0.5,
                text="Failed Fit",
                axref="x domain",
                ayref="y domain",
                ax=0.5,
                ay=0.5,
                arrowhead=2,
                font={"size": 20, "color": "Red"},
            )
            return fig

    def plot_full_procedure(
        self,
    ):
        twoD_with_centres = self.plot_2d_with_centers()
        outlier_thetas = self.plot_outlier_thetas()
        outlier_transition = self.plot_transition_outliers()
        avg_data = self.plot_avg_data()

        # get others
        full_fig = figures_to_subplots(
            [twoD_with_centres, outlier_thetas, outlier_transition, avg_data],
            rows=2,
            cols=2,
        )

        return full_fig

