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
from typing import Any, Union
import plotly.io as pio
import unicodedata
import datetime

from dat_analysis.dat.dat_hdf import DatHDF 
from dat_analysis.dat.logs_attr import Logs
from dat_analysis.plotting.plotly.util import default_fig, heatmap, error_fill, figures_to_subplots, fig_waterfall, limit_max_datasize
from dat_analysis.useful_functions import get_matching_x, bin_data, decimate, center_data, mean_data, resample_data, get_data_index, ensure_list
from dat_analysis.analysis_tools.data_aligning import subtract_data, subtract_data_1d
from dat_analysis.hdf_util import NotFoundInHdfError


from dat_analysis.analysis_tools.general_fitting import are_params_equal, FitResult, SimultaneousFitResult


######################################
####### Additions to DatHDF ##########
######################################
@property
def fig_addon(self: DatHDF):
    """For adding to DatHDF object to make accessing a default fig layout easier
    Returns
        go.Figure: With default layout and xaxis_title, yaxis_title, title etc
    """
    fig = default_fig()
    fig.update_layout(
        xaxis_title=self.Logs.x_label,
        yaxis_title=self.Logs.y_label,
        title=f"Dat{self.datnum}: ",
    )
    return fig

def __new_dat_repr__(self):
    """Give a more useful summary of DatHDF e.g. when left at end of jupyter cell"""
    return f"Dat{self.datnum} - {self.Logs.time_completed}"


def _get_Data(self, key):
    """
        Get more complete Data object directly from Dat
        
        I.e. including x (and y) axes, x_label, y_label, title (with datnum)
    """
    keys = self.Data._get_all_data_keys()
    if key in keys:
        data = self.Data._load_data(key)
    else:
        raise NotFoundInHdfError(
            f"{key} not found. Existing keys are {self.Data.data_keys}"
        )
    data = Data(
        data=data,
        x=self.Data.x,
        y=self.Data.y,
        plot_info=PlottingInfo(
            x_label=self.Logs.x_label,
            y_label=self.Logs.y_label,
            title=f"Dat{self.datnum}: {key}",
        ),
    )
    return data


def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')




@dataclass(frozen=True)
class FigInfo:
    filename: str
    time_saved: pd.Timestamp = field(compare=False)
    title: str
    trace_type: str
    datashape: Union[tuple[int], None]
    hdf_path: str = field(default=None, compare=False)
    fig: Optional[go.Figure] = field(default=None, compare=False)
    
    @classmethod
    def from_fig(cls, fig: go.Figure, filename: str = None) -> FigInfo:
        info = {}
        if filename is None:
            if fig.layout.title.text:
                filename = fig.layout.title.text 
            else:
                raise ValueError(f'If the fig has no title, a filename must be passed, got neither.')
                
        # Make filename filename safe
        filename = slugify(filename, allow_unicode=True)
        info['filename'] = filename
        info['time_saved'] = pd.Timestamp.now()
        info['title'] = fig.layout.title.text if fig.layout.title.text else ''
        info['trace_type'] = str(type(fig.data[0])) if fig.data else ''
        if fig.data:
            trace = fig.data[0]
            data = getattr(trace, 'z', None)
            if data is None:
                data = getattr(trace, 'y', None)
            if data is not None:
                info['datashape'] = tuple(data.shape)
        info['fig'] = fig
        return cls(**info)
        
    @classmethod
    def from_group(cls, group: h5py.Group, load_fig=True) -> FigInfo:
        """Read from group to build instance of FigInfo
        Note: Read only!
        """
        if group.attrs.get('dataclass', None) != 'FigInfo':
            raise NotFoundInHdfError(f'{group.name} is not a saved FigInfo')
            
        info = {
            k: group.attrs.get(k, None) for k in ['filename', 'title', 'trace_type']
        }
        info['time_saved'] = pd.Timestamp(group.attrs['time_saved'])
        info['datashape'] = tuple(group.attrs.get('datashape', tuple()))
        if load_fig:
            info['fig']= pio.from_json(group.attrs['fig'])
        info['hdf_path'] = group.file.filename
        return cls(**info)
        
    def to_group(self, parent_group: h5py.Group, overwrite=False):
        """Write to group everything that is needed to load again
        Note: Write allowed. Try not to write big things if not necessary (I think HDFs do not reclaim reused space)
        """
        if not self.fig:
            raise ValueError(f'This FigInfo does not contain a go.Figure (`self.fig = None`). Not saving')
        if overwrite:
            if self.filename in parent_group.keys():
                logging.info(f'Overwriting {self.filename} in {parent_group.file.filename}')
                del parent_group[self.filename]
        group = parent_group.require_group(self.filename)
        if group.attrs.get('dataclass', None) != 'FigInfo':
            group.attrs['dataclass'] = 'FigInfo'
        for k in ['filename', 'title', 'trace_type']:
            if group.attrs.get(k, None) != getattr(self, k) or overwrite:
                group.attrs[k] = getattr(self, k)
        
        if pd.Timestamp(group.attrs.get('time_saved', None)) != self.time_saved:
            group.attrs['time_saved'] = str(self.time_saved)
        
        if tuple(group.attrs.get('datashape', tuple())) != self.datashape:
            group.attrs['datashape'] = self.datashape
        
        fig_json = self.fig.to_json()
        if group.attrs.get('fig', None) != fig_json:
            group.attrs['fig'] = '' if self.fig is None else fig_json
        return self
        
    
@dataclass
class FigInfos:
    infos: tuple[FigInfo]
    latest_fig: FigInfo
    
    @property
    def filenames(self) -> list[str]:
        return [info.filename for info in self.infos]
    
    @classmethod
    def from_group(cls, fig_group: h5py.Group, load_figs=True) -> FigInfos:
        """
        Get all FigInfos saved in HDF group (optionally there info only if load_figs is False (fast))
        Note: Read only!
        """
        fig_infos = []
        for k in fig_group.keys():
            single_fig_group = fig_group[k]
            if single_fig_group.attrs.get('dataclass', None) == 'FigInfo':
                fig_info = FigInfo.from_group(single_fig_group, load_fig=load_figs)
                fig_infos.append(fig_info)
        if not fig_infos:
            return None
        
        # Order newest first
        fig_infos = tuple(reversed(sorted(fig_infos, key=lambda info: info.time_saved)))
        latest_fig = fig_infos[0]
        inst = cls(fig_infos, latest_fig)
        return inst
    
    
def _save_fig(self, fig: go.Figure, filename: str=None, overwrite=False) -> FigInfo:
    """Save Figure to HDF
    Args:
        filename: optionally provide the name to store under (defaults to fig title)
    """
    assert isinstance(fig, go.Figure)
    fig_info = FigInfo.from_fig(fig, filename=filename)
   
    if not overwrite:
        # Avoid entering write mode if not necessary
        existing = self._load_fig_info(filename=fig_info.filename, load_fig=False)
        if existing == fig_info:
            logging.info(f'Fig ({fig_info.filename}) already saved in Dat{self.datnum}, to overwrite, set `overwrite` = True')
            return fig_info
        elif existing:
            logging.info(f'Ovewriting Fig ({fig_info.filename}) in Dat{self.datnum}. Existing fig had same title but was different')
    # Write fig to HDF
    with self.hdf_write as f:
        fig_group = f.require_group('Figs')
        fig_info.to_group(parent_group=fig_group, overwrite=overwrite)
    logging.info(f'Fig ({fig_info.filename}) saved in Dat{self.datnum}')
    return fig_info

def _saved_figs(self, load_figs=True) -> Optional[FigInfos]:
    """Load saved Figs from HDF
    Args:
        load_figs: If False will load the attrs only (fast), otherwise will also load the full figure
    """
    fig_infos = None 
    with self.hdf_read as f:
        if 'Figs' in f.keys():
            fig_infos = FigInfos.from_group(f['Figs'], load_figs=load_figs)
    return fig_infos

def _load_fig_from_hdf(self, filename, load_fig=True) -> FigInfo:
    with self.hdf_read as f:
        if 'Figs' in f.keys():
            fig_group = f['Figs']
            if filename in fig_group.keys():
                fig_info = FigInfo.from_group(fig_group[filename], load_fig=load_fig)
                return fig_info
    return None
    

def _load_fig_info(self, filename: str = None, load_fig=True) -> Optional[FigInfo]:
    """Load the named fig from HDF, or if None, the last saved fig"""
    # Get Info on Saved Figs
    saved = self.saved_figs(load_figs=False)
    
    # If nothing saved, return None
    if not saved or not saved.infos:
        return None
    
    # If no filename specified, get latest saved fig
    filename = filename if filename else saved.latest_fig.filename
    
    # If present, load it
    if filename in saved.filenames:
        fig_info = self._load_fig_from_hdf(filename, load_fig=load_fig)
        return fig_info
    return None

def _load_fig(self, filename: str = None) -> Optional[go.Figure]:
    """Load the named fig from HDF, or if None, the last saved fig"""
    fig_info = self._load_fig_info(filename=filename)
    if fig_info:
        return fig_info.fig
    return None
    
    
def _plot(self, fig: go.Figure, overwrite=False) -> go.Figure:
    """Adds extra info to figure from Dat file and also saves the figure to the HDF"""
    # TODO: Decide where to put "Saved in..." annotation based on Figure height (for larger height, smaller number works better)
    if not np.any([annotation.y == -0.15 for annotation in fig.layout.annotations]):
        fig.add_annotation(
            xref="paper", yref="paper", x=1.0, y=-0.15, text=f"Saved in Dat{self.datnum}", showarrow=False
        )
    self.save_fig(fig, overwrite=overwrite)
    return fig
        

# Monkey patch to DatHDF class
DatHDF.standard_fig = fig_addon
DatHDF.__repr__ = __new_dat_repr__
DatHDF.get_Data = _get_Data
DatHDF.save_fig = _save_fig
DatHDF._load_fig_from_hdf = _load_fig_from_hdf
DatHDF._load_fig_info = _load_fig_info
DatHDF.load_fig = _load_fig
DatHDF.saved_figs = _saved_figs
DatHDF.plot = _plot


######################################
####### End Additions to DatHDF ######
######################################

######################################
####### Additions to Logs ############
######################################
def scan_vars(self):
    """Get the dictionary of Scan Vars from HDF File"""
    with self.hdf_read as f:
        scan_vars = json.loads(f["Logs"].attrs["scan_vars_string"])
    return scan_vars

@dataclass(frozen=True)
class AxisGates:
    dacs: tuple[int]
    channels: tuple[str]
    starts: tuple[float]
    fins: tuple[float]
    numpts: int

    def to_df(self) -> pd.DataFrame:
        df = pd.DataFrame(
            [self.starts, self.fins, self.dacs],
            columns=self.channels,
            index=["starts", "fins", "dac"],
        )
        return df

    def values_at(self, value: float, channel: str = None) -> dict[str, float]:
        """Return dict of DAC values at given value of channel
        Args:
            value: DAC value of channel to evaluate other DAC values at 
            channel: Which channel the value corresponds to (by default the first channel)
        """
        channel = channel if channel else self.channels[0]
        index = self.channels.index(channel)
        start = self.starts[index]
        fin = self.fins[index]

        proportion = (value - start) / (fin - start)
        return self.calculate_axis_gate_vals_from_sweep_proportion(proportion)
        
    def calculate_axis_gate_vals_from_sweep_proportion(self,
        proportion
    ) -> dict[str, float]:
        """
        Return dict of DAC values at proportion along sweep
        
        Args:
            proportion: Proportion of sweep to return values for (e.g. 0.0 is start of sweep, 1.0 is end of sweep)
        """
        return {
            k: s + proportion * (f - s)
            for k, s, f in zip(self.channels, self.starts, self.fins)
    }

@dataclass(frozen=True)
class SweepGates:
    x: AxisGates = None
    y: AxisGates = None

    def plot(self, axis: str ='x', numpts=200):
        fig = default_fig()
        df = getattr(self, axis).to_df()
        main_gate = df.T.iloc[0]
        main_x = np.linspace(main_gate.starts, main_gate.fins, numpts)
        for name, row in df.T.iterrows():
            y = np.linspace(row.starts, row.fins, numpts)
            fig.add_trace(go.Scatter(x=main_x, y=y, name=name))
        fig.update_layout(
            title=f"{axis} axis sweepgates",
            xaxis_title=f"Main sweep gate ({main_gate.name}) /mV",
            yaxis_title="Other Gate Values /mV",
            hovermode="x unified",
        )
        return fig 
    
    def convert_to_real(self):
        """If gates have voltage dividers and are named e.g. "P*200", this will return the real applied voltages and gate names"""
        return convert_sweepgates_to_real(self)
    
    
def get_sweepgates(self):
    scan_vars = self.scan_vars
    axis_gates = {}
    for axis in ["x", "y"]:
        starts = scan_vars.get(f"start{axis}s")
        fins = scan_vars.get(f"fin{axis}s")
        if starts == "null" or fins == "null":
            continue
        starts, fins = [
            tuple([float(v) for v in vals])
            for vals in [starts.split(","), fins.split(",")]
        ]
        channels = scan_vars.get(f"channels{axis}")
        channels = tuple([int(v) for v in channels.split(",")])
        numpts = scan_vars.get(f"numpts{axis}")
        channel_names = scan_vars.get(f"{axis}_label")
        if channel_names.endswith(" (mV)"):
            channel_names = channel_names[:-5]
        else:
            channel_names = ",".join([f"DAC{num}" for num in channels])
        channel_names = tuple([name.strip() for name in channel_names.split(",")])
        axis_gates[axis] = AxisGates(channels, channel_names, starts, fins, numpts)
    return SweepGates(**axis_gates)


# Monkey patch to Logs class
Logs.scan_vars = property(scan_vars)
Logs.sweepgates = property(get_sweepgates)




def _dividers_from_gate_names(gate_names) -> np.ndarray:
    dividers = [
        float(re.search("\*(\d+)", gate_name).groups()[0]) for gate_name in gate_names
    ]
    return np.array(dividers)


def _gate_names_excluding_dividers(gate_names) -> list[str]:
    gates = [re.search("(.*)\*", gate_name).groups()[0] for gate_name in gate_names]
    return gates


def convert_sweepgates_to_real(sweepgates: SweepGates) -> SweepGates:
    """If gates have voltage dividers and are named e.g. "P*200", this will return the real applied voltages and gate names"""
    real_axis_gates = {}
    for axis in ["x", "y"]:
        axis_gates = getattr(sweepgates, axis)
        if axis_gates is not None:
            dividers = _dividers_from_gate_names(axis_gates.channels)
            real_starts = axis_gates.starts / dividers
            real_fins = axis_gates.fins / dividers
            real_channels = _gate_names_excluding_dividers(axis_gates.channels)
            real_axis_gates[axis] = AxisGates(
                dacs=axis_gates.dacs,
                channels=real_channels,
                starts=real_starts,
                fins=real_fins,
                numpts=axis_gates.numpts,
            )
    return SweepGates(**real_axis_gates)


    

######################################
####### End Additions to Logs ########
######################################
    
#######################################################
########### Data class stuff  #########################
#######################################################
from dat_analysis.analysis_tools.data import PlottingInfo, Data
# @dataclass
# class PlottingInfo:
#     x_label: str = None
#     y_label: str = None
#     title: str = None
#     coloraxis_title: str = None
#     datnum: int = None  # TODO: Remove

#     @classmethod
#     def from_dat(cls, dat: DatHDF, title: str = None):
#         inst = cls(
#             x_label=dat.Logs.x_label,
#             y_label=dat.Logs.y_label,
#             title=f"Dat{dat.datnum}: {title}",
#             datnum=dat.datnum,
#         )
#         return inst
    
#     def update_layout(self, fig: go.Figure):
#         """Updates layout of figure (only with non-None values)"""
#         updates = {
#             k: v
#             for k, v in {
#                 "xaxis_title_text": self.x_label,
#                 "yaxis_title_text": self.y_label,
#                 "title_text": self.title,
#                 "coloraxis_colorbar_title_text": self.coloraxis_title,
                
#             }.items()
#             if v is not None
#         }
#         return fig.update_layout(**updates)
    
# @dataclass
# class Data:
#     data: np.ndarray
#     x: np.ndarray
#     y: np.ndarray = None
#     xerr: np.ndarray = None
#     yerr: np.ndarray = None
#     plot_info: PlottingInfo = PlottingInfo()
    
#     def __post_init__(self, *args, **kwargs):
#         # If data is 2D and no y provided, just assign index values
#         self.x = np.asanyarray(self.x) if self.x is not None else self.x
#         self.y = np.asanyarray(self.y) if self.y is not None else self.y
#         self.data = np.asanyarray(self.data) if self.data is not None else self.data
#         if self.data.ndim == 2 and self.y is None:
#             self.y = np.arange(self.data.shape[0])
#         if self.plot_info is None:
#             self.plot_info = PlottingInfo()
        
#         if args:
#             logging.warning(f'Data got unexpected __post_init__ args {args}')
#         if kwargs:
#             logging.warning(f'Data got unexpected __post_init__ kwargs {kwargs}')
            
#     def plot(self, limit_datapoints=True, **trace_kwargs):
#         """
#             Generate a quick 1D or 2D plot of data
#             Args:
#                 limit_datapoints: Whether to do automatic downsampling before plotting (very useful for large 2D datasets)
#         """
#         fig = default_fig()
#         if self.data.ndim == 1:
#             fig.add_traces(self.get_traces(**trace_kwargs))
#         elif self.data.ndim == 2:
#             if limit_datapoints:
#                 fig.add_trace(heatmap(x=self.x, y=self.y, data=self.data, **trace_kwargs))
#             else:
#                 fig.add_trace(go.Heatmap(x=self.x, y=self.y, z=self.data, **trace_kwargs))
#         else:
#             raise RuntimeError(f"data is not 1D or 2D, got data.shape = {self.data.ndim}")
#         if self.plot_info:
#             fig = self.plot_info.update_layout(fig)
#         return fig
    
#     def get_traces(self, max_num_pnts=10000, error_fill_threshold=50, **first_trace_kwargs) -> list[Union[go.Scatter, go.Heatmap]]:
#         traces = []
#         group_key = first_trace_kwargs.pop('legendgroup', uuid.uuid4().hex)
#         if self.data.ndim == 1:
#             if max_num_pnts:
#                 data, x = resample_data(self.data, x=self.x, max_num_pnts=max_num_pnts, resample_method='downsample', resample_x_only=True)
#             # Note: Scattergl might cause issues with lots of plots, if so just go back to go.Scatter only
#             # scatter_func = go.Scatter if x.shape[0] < 1000 else go.Scattergl  
#             scatter_func = go.Scatter
#             trace = scatter_func(x=x, y=data, legendgroup=group_key, **first_trace_kwargs)
#             traces.append(trace)
#             if self.yerr is not None:
#                 yerr = resample_data(self.yerr,  max_num_pnts=max_num_pnts, resample_method='downsample', resample_x_only=True)
#                 if len(x) <= error_fill_threshold:
#                     trace.update(error_y=dict(array=yerr))
#                 else:
#                     # Note: error_fill also switched to scattergl after 1000 points
#                     traces.append(error_fill(x, data, yerr, legendgroup=group_key))
#             if self.xerr is not None:
#                 xerr = resample_data(self.xerr,  max_num_pnts=max_num_pnts, resample_method='downsample', resample_x_only=True)
#                 if len(x) <= error_fill_threshold:
#                     trace.update(error_x=dict(array=xerr), legendgroup=group_key)
#                 else:
#                     pass # Too many to plot, don't do anything
#         elif self.data.ndim == 2:
#             traces.append(heatmap(x=self.x, y=self.y, data=self.data, **first_trace_kwargs))
#         else:
#             raise RuntimeError(f"data is not 1D or 2D, got data.shape = {self.data.ndim}")
#         return traces
    
#     def copy(self) -> Data:
#         """Make a copy of self s.t. changing the copy does not affect the original data"""
#         return copy.deepcopy(self)
    
#     def center(self, centers):
#         centered, new_x = center_data(self.x, self.data, centers, return_x=True)
#         new_data = self.copy()
#         new_data.x = new_x
#         new_data.data = centered
#         return new_data

#     def mean(self, centers=None, axis=None):
#         axis = axis if axis else 0
#         if centers is not None:
#             if axis != 0:
#                 raise NotImplementedError
#             averaged, new_x, averaged_err = mean_data(self.x, self.data, centers, return_x = True, return_std=True)
#         else:
#             averaged, averaged_err = np.mean(self.data, axis=axis), np.std(self.data, axis=axis)
#             if self.data.ndim == 2:
#                 if axis == 0:
#                     new_x = self.x
#                 elif axis in [1, -1]:
#                     new_x  = self.y
#                 else:
#                     raise ValueError(f'axis {axis} not valid/implemented')
#             elif self.data.ndim == 1 and axis == 0:
#                 # Only return average value (not really Data anymore)
#                 return averaged
#             else:
#                 raise ValueError(f'axis {axis} not valid/implemented')
#         new_data = self.copy()
#         new_data.x = new_x
#         new_data.y = None
#         new_data.plot_info.y_label = new_data.plot_info.coloraxis_title
#         new_data.plot_info.title = f"{new_data.plot_info.title} Averaged in axis {axis}{' after aligning' if centers is not None else ''}"
#         new_data.data = averaged
#         new_data.yerr = averaged_err
#         return new_data
    
#     def __add__(self, other: Data) -> Data:
#         return self.add(other)
    
#     def __sub__(self, other: Data) -> Data:
#         return self.subtract(other)
    
#     def subtract(self, other_data: Data) -> Data:
#         new_data = self.copy()
#         if self.data.ndim == 1:
#             new_data.x, new_data.data = subtract_data_1d(self.x, self.data, other_data.x, other_data.data)
#         elif self.data.ndim == 2:
#             new_data.x, new_data.y, new_data.data = subtract_data(self.x, self.y, self.data, other_data.x, other_data.y, other_data.data)
#         else:
#             raise NotImplementedError(f"Subtracting for data with ndim == {self.data.ndim} not implemented")
#         return new_data
    
#     def add(self, other_data: Data) -> Data:
#         od = other_data.copy()
#         od.data = -1*od.data
#         return self.subtract(od)
    
#     def diff(self, axis=-1) -> Data:
#         """Differentiate Data long specified axis
        
#         Note: size reduced by 1 in differentiation axis
#         """
#         data = self.copy()
#         diff = np.diff(self.data, axis=axis)
#         data.data = diff
#         data.x = get_matching_x(self.x, shape_to_match=diff.shape[-1])
#         if self.y is not None:
#             data.y = get_matching_x(self.y, shape_to_match=diff.shape[0])
#         else:
#             data.y = None
#         return data

#     def smooth(self, axis=-1, window_length=10, polyorder=3) -> Data:
#         """Smooth data using method savgol_filter"""
#         data = self.copy()
#         data.data = savgol_filter(self.data, window_length, polyorder)
#         data.plot_info.title = f'{data.plot_info.title} Smoothed ({window_length})'
#         return data
    
#     def decimate(
#             self,
#             decimate_factor: int = None,
#             numpnts: int = None,
#             measure_freq: float = None,
#             desired_freq: float = None,
#         ):
#         """Decimate data (i.e. lowpass then downsample)
#         Note: Use either decimate_factor, numpnts, or (measure_freq and desired_freq)
#         """
#         data = self.copy()
#         data.data = decimate(
#             self.data.astype(float),
#             measure_freq=measure_freq,
#             desired_freq=desired_freq,
#             decimate_factor=decimate_factor,
#             numpnts=numpnts,
#         )
#         data.yerr = None  # I don't think this makes sense after decimating
#         data.x = get_matching_x(self.x.astype(float), shape_to_match=data.data.shape[-1])
#         data.plot_info.title = f'{data.plot_info.title} Decimated ({len(data.x)} points)'
#         return data

#     def notch_filter(
#         self,
#         notch_freq: Union[float, list[float]],
#         Q: float,
#         measure_freq: float,
#         fill_nan_values: float = None,
#     ):
#         notch_freqs = ensure_list(notch_freq)
#         data = self.copy()
#         if np.sum(np.isnan(data.data)) > 0:
#             if fill_nan_values is None:
#                 raise ValueError(
#                     f"Data must either contain no NaNs or `fill_nan_values` must be provided"
#                 )
#             data.data[np.isnan(data.data)] = fill_nan_values
            
#         for notch_freq in notch_freqs:
#             b, a = iirnotch(notch_freq, Q, fs=measure_freq)
#             data.data = filtfilt(b, a, data.data)
#         data.plot_info.title = f'{data.plot_info.title} Notch Filtered ({notch_freqs} Hz)'
#         return data
    
#     def bin(self, bin_x=1, bin_y=1) -> Data:
#         """Bin data
#         Args:
#             bin_x: binsize for x-axis
#             bin_y: binsize for y-axis
#         """
#         data = self.copy()
#         data.data = bin_data(self.data, bin_x=bin_x, bin_y=bin_y)
#         data.x = bin_data(self.x, bin_x=bin_x)
#         if data.y is not None:
#             data.y = bin_data(self.y, bin_x=bin_y)
#         return data
    
#     def __getitem__(self, key: tuple):
#         """Allow slicing like a numpy array"""
#         new_data = self.copy()

#         new_data.data = self.data[key]
#         if new_data.yerr is not None:
#             new_data.yerr = self.yerr[key]

#         if self.data.ndim == 2:
#             if isinstance(key, tuple) and len(key) == 2:
#                 pass
#             elif isinstance(key, slice):
#                 key = tuple([key, ...])
#             elif isinstance(key, int):
#                 key = tuple([key, ...])
#             else:
#                 raise NotImplementedError
#             new_data.x = self.x[key[1]]
#             new_data.y = self.y[key[0]]
#         elif self.data.ndim == 1:
#             new_data.x = self.x[key]
#         else:
#             raise NotImplementedError
#         return new_data

#     def slice_values(
#         self,
#         x_range: tuple[Optional[int], Optional[int]] = None,
#         y_range: tuple[Optional[int], Optional[int]] = None,
#     ):
#         """Select data based on x and y axes"""
#         def replace_nones_in_indexes(arr:np.ndarray, indexes: tuple):
#             """Replace None with either 0 or last index value"""
#             new_indexes = list(indexes)
#             if indexes[0] is None:
#                 indexes[0] = 0
#             if indexes[1] is None:
#                 indexes[1] = arr.shape[-1]
#             return indexes
        
#         x_slice = ...
#         if x_range is not None:
#             x_indexes = get_data_index(self.x, x_range)
#             x_indexes = replace_nones_in_indexes(self.x, x_indexes)
#             if None not in x_indexes:
#                 x_indexes = min(x_indexes), max(x_indexes)
#             x_slice = slice(x_indexes[0], x_indexes[1] + 1)
#         if self.data.ndim == 1:
#             return self[x_slice]
#         elif self.data.ndim == 2:
#             y_slice = ...
#             if y_range is not None:
#                 y_indexes = get_data_index(self.y, y_range)
#                 y_indexes = replace_nones_in_indexes(self.y, y_indexes)
#                 if None not in y_indexes:
#                     y_indexes = min(y_indexes), max(y_indexes)
#                 y_slice = slice(y_indexes[0], y_indexes[1] + 1)
#             return self[y_slice, x_slice]
#         else:
#             raise NotImplementedError
    
#     def _ipython_display_(self):
#         """Make this object act like a figure when calling display(data) or leaving at the end of a jupyter cell"""
#         return self.plot()._ipython_display_()
    
    
from dat_analysis.analysis_tools.data import InterlacedData

# @dataclass
# class InterlacedData(Data):
#     """E.g. Combining +/- conductance data or averaging the same bias CS data"""

#     num_setpoints: int = None

#     def __post_init__(self, *args, **kwargs):
#         super().__post_init__(*args, **kwargs)
#         if self.num_setpoints is None:
#             raise ValueError(f"must specify `num_setpoints` for InterlacedData")
    
#     @classmethod
#     def from_Data(cls, data: Data, num_setpoints: int):
#         """Convert a regulare Data class to InterlacedData"""
#         d = data.copy()
#         if isinstance(d, cls):
#             # Already InterlacedData, just update num_setpoints
#             d.num_setpoints = num_setpoints
#             return d
#         inst = cls(**d.__dict__, num_setpoints=num_setpoints)
#         return inst

#     @classmethod
#     def get_num_interlaced_setpoints(cls, scan_vars) -> int:
#         """
#             Helper function to not have to remember how to do this every time
#             Returns:
#                 (int): number of y-interlace setpoints
#         """
#         if scan_vars.get("interlaced_y_flag", 0):
#             num = len(scan_vars["interlaced_setpoints"].split(";")[0].split(","))
#         else:
#             num = 1
#         return num
    
#     def separate_setpoints(self) -> list[Data]:
#         new_y = np.linspace(
#             self.y[0],
#             self.y[-1],
#             int(self.y.shape[0] / self.num_setpoints),
#         )
#         new_datas = []
#         for i in range(self.num_setpoints):
#             d_ = copy.deepcopy(self.__dict__)
#             d_.pop('num_setpoints')
#             new_data = Data(**d_)
#             new_data.plot_info.title = f'Interlaced Data Setpoint {i}'  
#             new_data.y = new_y
#             new_data.data = self.data[i :: self.num_setpoints]
#             new_datas.append(new_data)
#         return new_datas
    
#     def combine_setpoints(
#         self, setpoints: list[int], mode: str = "mean", centers=None
#     ) -> Data:
#         """
#         Combine separate parts of interlaced data by averaging or difference

#         Args:
#             setpoints: which interlaced setpoints to combine
#             mode: `mean` or `difference`
#         """
#         if mode not in (modes := ["mean", "difference"]):
#             raise NotImplementedError(f"{mode} not implemented, must be in {modes}")

#         if centers is not None:
#             data = self.center(centers)
#         else:
#             data = self
#         datas = np.array(data.separate_setpoints())[list(setpoints)]
#         new_data = datas[0].copy()
#         if mode == "mean":
#             new_data.data = np.nanmean([data.data for data in datas], axis=0)
#         elif mode == "difference":
#             if len(datas) != 2:
#                 raise ValueError(
#                     f"Got {setpoints}, expected 2 exactly for mode `difference`"
#                 )
#             new_data.data = datas[0].data - datas[1].data
#         else:
#             raise RuntimeError
#         new_data.plot_info.title = f'Combined by {mode} of {setpoints}'
#         return new_data

#     def plot_separated(self, shared_data=False) -> go.Figure:
#         figs = []
#         for i, d in enumerate(self.separate_setpoints()):
#             fig = default_fig()
#             fig.add_trace(heatmap(d.x, d.y, d.data))
#             fig.update_layout(title=f"Interlace Setpoint: {i}")
#             figs.append(fig)

#         title = self.plot_info.title if self.plot_info.title else 'Separated Data'
#         fig = figures_to_subplots(
#             figs,
#             title=title,
#             shared_data=shared_data,
#         )
#         if self.plot_info:
#             fig = self.plot_info.update_layout(fig)
#             fig.update_layout(title=f'{fig.layout.title.text} Interlaced Separated')
#             # Note: Only updates xaxis1 by default, so update other axes
#             fig.update_xaxes(title=self.plot_info.x_label)
#             fig.update_yaxes(title=self.plot_info.y_label)
#         return fig

#     def center(self, centers) -> InterlacedData:
#         """If passed a list of list of centers, flatten back to apply to whole dataset before calling super().center(...)"""
#         if len(centers) == self.num_setpoints:
#             # Flatten back to a single center per row for the whole dataset
#             centers = np.array(centers).flatten(order="F")
#         return super().center(centers)
    
    
#     def _ipython_display_(self):
#         """Make this object act like a figure when calling display(data) or leaving at the end of a jupyter cell"""
#         return self.plot_separated()._ipython_display_()
    


######################  End of Data Stuff ######################
    
    
#######################################################
########### Plotting Updates ##########################
#######################################################



def error_fill(x, data, error, **kwargs):
    # NOTE: This is a copy from dat_analysis.plotting.plotly.util>error_fill 
    # This is here just to try using Scattergl for large plots
    if isinstance(error, numbers.Number):
        error = [error] * len(x)
    x, data, error = [np.array(arr) for arr in [x, data, error]]
    upper = data + error
    lower = data - error

    fill_color = kwargs.pop('fill_color', 'rgba(50, 50, 50, 0.2)')
    # Note: Scattergl might cause issues with lots of plots, if so just go back to go.Scatter only
    # scatter_func = go.Scatter if len(x) < 1000 else go.Scattergl  
    scatter_func = go.Scatter
    return scatter_func(
        x=np.concatenate((x, x[::-1])),
        y=np.concatenate((upper, lower[::-1])),
        fill='tozeroy',
        fillcolor=fill_color,
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False,
        **kwargs,
    )
    
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
