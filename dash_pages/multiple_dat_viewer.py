from __future__ import annotations
from typing import TYPE_CHECKING
import pandas as pd
from dictor import dictor
import textwrap
import dash
import logging

import dat_analysis.dash.util as du
from dat_analysis.dash.util import Components as C, make_app
from dat_analysis.plotting.plotly.util import figures_to_subplots, default_fig

from default_import import get_dat

if TYPE_CHECKING:
    from dat_analysis.dat.dat_hdf import DatHDF

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(funcName)s:%(message)s")


def single_dat_general_info(dat: DatHDF) -> dict:
    sl = dat.Logs.sweeplogs
    d = {}
    d["Time Completed"] = pd.Timestamp(dictor(sl, "time_completed", None))
    d["Magnet (mT)"] = dictor(sl, "LS625 Magnet Supply.field mT", None)
    d["Time Elapsed"] = dictor(sl, "time_elapsed", None)
    d["Measure Freq"] = dictor(sl, "measureFreq", None)
    d["Resample Freq"] = dictor(sl, "resamplingFreq", None)
    d["X Label"] = dictor(sl, "axis_labels.x", None)
    d["Y Label"] = dictor(sl, "axis_labels.y", None)
    return d


# Make App
app = make_app()

# Make stores (place to store output of function
stores = [
    store_datnums := C.Store(),
    store_dats := C.Store(),
    store_dacs_df := C.Store(),
    store_temps_df := C.Store(),
    store_general_df := C.Store(),
]
#  Make outputs (to show the output in dash app)
outputs = [
    table_dacs := C.DataTable("table-dacs", pd.DataFrame(), display_name="Dacs"),
    table_temps := C.DataTable(
        "table-temps", pd.DataFrame(), display_name="Temperatures"
    ),
    table_general := C.DataTable(
        "table-general", pd.DataFrame(), display_name="General Info"
    ),
    div_2d_graphs := C.Div("div-graphs-2d", header="2D Data"),
    div_1d_graphs := C.Div("div-graphs-1d", header="1D Data"),
]

# Make inputs (to fill values the function takes)
inputs = [
    in_datnums := C.Input(
        "Datnums (comma separated)",
        debounce=True,
        autofocus=True,
        inputmode="latin",
        placeholder="E.G. 100, 250, 1000",
        persistence=True,
        persistence_type="session",
    ),
]


##### Store Callbacks
# app.callback(
#     output=.as_output(serverside=True),
#     inputs=.as_input(),
# )(lambda : )
@app.callback(
    output=store_datnums.as_output(serverside=False),
    inputs=[in_datnums.as_input(), store_datnums.as_state()],
)
def update_datnums(datnum_input, previous):
    logging.info(f"got {datnum_input}")
    try:
        datnums = [int(v.strip(" ")) for v in datnum_input.split(",")]
    except (ValueError, AttributeError):
        datnums = []

    if datnums != previous:
        return datnums
    else:
        return dash.no_update


@app.callback(
    output=store_dats.as_output(serverside=True),
    inputs=[store_datnums.as_input(), store_dats.as_state()],
)
def update_dats(datnums, previous) -> dict[int, DatHDF]:
    # To be able to use previously loaded dats without loading again
    new_dats = {}
    for num in datnums:
        if previous and num in previous:
            new_dats[num] = previous[num]
        else:
            try:
                dat = get_dat(num)
                new_dats[num] = dat
            except FileNotFoundError:
                logging.warning(f"Dat{num} not found")
                pass
    return new_dats


@app.callback(
    output=store_dacs_df.as_output(serverside=True),
    inputs=[store_dats.as_input()],
)
def update_dacs_df(dats_dict):
    if dats_dict:
        dacs_dict = {num: dat.Logs.dacs for num, dat in dats_dict.items()}
        dacs_df = pd.DataFrame(
            dacs_dict,
        ).T
        dacs_df.index.name = "Datnum"
        return dacs_df
    return pd.DataFrame()


@app.callback(
    output=store_temps_df.as_output(serverside=True),
    inputs=[store_dats.as_input()],
)
def update_temps_df(dats_dict):
    if dats_dict:
        temps_dict = {
            num: dat.Logs.temperatures.asdict() for num, dat in dats_dict.items()
        }
        temps_df = pd.DataFrame(temps_dict).T
        temps_df.index.name = "Datnum"
        return temps_df
    return pd.DataFrame()


@app.callback(
    output=store_general_df.as_output(serverside=True),
    inputs=[store_dats.as_input()],
)
def update_general_df(dats_dict):
    if dats_dict:
        general_dict = {
            num: single_dat_general_info(dat) for num, dat in dats_dict.items()
        }
        general_df = pd.DataFrame(general_dict).T
        general_df.index.name = "Datnum"
        return general_df
    return pd.DataFrame()


##### Input Callbacks (i.e. update Dropdown options)
# app.callback(
#     output=[.as_output('options'), .as_output('value')],
#     inputs=.as_input(),
# )(lambda opts: list(opts), opts[0])

##### Output Callbacks (also define any extra funcitons required)
# C.Graph.run_callbacks(app)
# app.callback(
#     output=.as_output(),
#     inputs=[.as_input(), .as_input()]
# )()
C.Graph.run_callbacks(app)

app.callback(
    output=[
        table_dacs.as_output("data"),
        table_dacs.as_output("columns"),
        table_dacs.as_output("style_data_conditional"),
    ],
    inputs=[store_dacs_df.as_input()],
)(
    lambda df: (
        C.DataTable.df_to_data(df),
        C.DataTable.df_to_columns(df),
        du.generate_conditional_format_styles(
            df, n_bins=5, columns="all", mode="per_column"
        ),
    )
)

app.callback(
    output=[
        table_temps.as_output("data"),
        table_temps.as_output("columns"),
        table_temps.as_output("style_data_conditional"),
    ],
    inputs=[store_temps_df.as_input()],
)(
    lambda df: (
        C.DataTable.df_to_data(df),
        C.DataTable.df_to_columns(df),
        du.generate_conditional_format_styles(
            df, n_bins=5, columns="all", mode="per_column"
        ),
    )
)

app.callback(
    output=[
        table_general.as_output("data"),
        table_general.as_output("columns"),
        table_general.as_output("style_data_conditional"),
    ],
    inputs=[store_general_df.as_input()],
)(
    lambda df: (
        C.DataTable.df_to_data(df),
        C.DataTable.df_to_columns(df),
        du.generate_conditional_format_styles(
            df, n_bins=5, columns="all", mode="per_column"
        ),
    )
)


@app.callback(
    output=div_2d_graphs.as_output(),
    inputs=[store_dats.as_input()],
)
def update_2d_graphs_div(dats_dict):
    if not dats_dict:
        return dash.html.Div(f"Nothing to display")

    data_figs = {}  # List of figs by data_key
    for datnum, dat in dats_dict.items():
        for k in dat.Data.data_keys:
            if k not in data_figs:
                data_figs[k] = []
            data = dat.get_Data(k)
            if data.data.ndim == 2:
                fig = data.plot(limit_datapoints=False)
                data_figs[k].append(fig)

    graphs = []
    for k, figs in data_figs.items():
        if len(figs) > 9:
            logging.warning(f"Plotting more than 9 2D datasets not implemented")
            return dash.html.Div(
                f"Plotting more than 9 2D datasets not implemented, got {len(figs)} figs"
            )

        full_fig = figures_to_subplots(
            data_figs[k], title=f"{k} Data for selected Dats", shared_data=False
        )
        graphs.append(C.Graph(figure=full_fig))
    return dash.html.Div(graphs)


@app.callback(
    output=div_1d_graphs.as_output(),
    inputs=[store_dats.as_input()],
)
def update_1d_graphs_div(dats_dict):
    if not dats_dict:
        return dash.html.Div(f"Nothing to display")

    datas_1d = {}  # List of figs by data_key
    for datnum, dat in dats_dict.items():
        for k in dat.Data.data_keys:
            if k not in datas_1d:
                datas_1d[k] = []
            data = dat.get_Data(k)
            if data.data.ndim == 2:
                data = data.mean(axis=0)
            datas_1d[k].append(data)

    graphs = []
    for k, datas in datas_1d.items():
        fig = default_fig()
        if datas[0].plot_info:
            datas[0].plot_info.update_layout(fig)
        fig.update_layout(title=f"{k} Data for selected Dats")
        for data in datas:
            name = "<br>".join(textwrap.wrap(data.plot_info.title, width=30))
            fig.add_traces(data.get_traces(name=name))
        graphs.append(C.Graph(figure=fig))
    return dash.html.Div(graphs)


# Put those components together into a layout
layout = du.make_layout_section(stores=stores, inputs=inputs, outputs=outputs)

# Make app and attach callbacks (Note: cannot re-run callbacks with existing app)
app.layout = layout
app.title = "Multiple Dat Viewer"

# Run the app
port = 9100
port = port if port else du.get_unused_port()
app.run(port=port, debug=True, dev_tools_hot_reload=True)
