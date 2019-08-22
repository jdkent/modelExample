''' Present an interactive function explorer with slider widgets.
Scrub the sliders to change the properties of the ``hrf`` curve, or
type into the title text box to update the title of the plot.
Use the ``bokeh serve`` command to run the example by executing:
    bokeh serve sliders.py
at your command prompt. Then navigate to the URL
    http://localhost:5006/sliders
in your browser.
'''
import numpy as np
from nistats import hemodynamic_models
from statsmodels.tsa.arima_process import ArmaProcess
import pandas as pd

from bokeh.layouts import grid
from bokeh.core.properties import value
from bokeh.palettes import Dark2
from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, Label, Div
from bokeh.models.widgets import Slider
from bokeh.plotting import figure


def generate_signal(go_onset=2, ss_onset=12, fs_onset=22,
                    go_pwr=1, ss_pwr=2, fs_pwr=3, noise=0,
                    design_resolution=0.1, duration=40,
                    stim_duration=1, tr=1):
    rho = 0.12
    cond_order = [0, 1, 2]
    betas = np.array([go_pwr, ss_pwr, fs_pwr])
    onsets = np.array([go_onset, ss_onset, fs_onset])
    onsets_res = onsets / design_resolution
    onsets_res = onsets_res.astype(int)
    duration_res = int(duration / design_resolution)
    stim_duration_res = int(stim_duration / design_resolution)
    sampling_rate = int(tr / design_resolution)

    X = np.zeros((duration_res, onsets.shape[0]))
    B = np.zeros((onsets.shape[0], 1))

    for idx, (cond, onset) in enumerate(zip(cond_order, onsets_res)):
        # set the design matrix
        X[onset:onset+stim_duration_res, idx] = 1
        X[:, idx] = np.convolve(
            X[:, idx], hemodynamic_models._gamma_difference_hrf(
                tr, oversampling=sampling_rate))[0:X.shape[0]]
        # set the beta for the trial depending on condition
        B[idx, :] = betas[cond]

    # downsample X so it's back to TR resolution
    X = X[::sampling_rate, :]

    signal = X @ B
    signal = np.squeeze(signal)
    if noise > 0.0:
        np.random.seed(12345)
        # make the noise component
        n_trs = int(duration / tr)
        ar = np.array([1, -rho])  # statmodels says to invert rho
        ap = ArmaProcess(ar)
        err = ap.generate_sample(n_trs, scale=noise, axis=0)

        Y = signal + err
    else:
        Y = signal

    return Y


# Set up data
duration = 40
Y = generate_signal()
go_est = generate_signal(go_pwr=1, ss_pwr=0, fs_pwr=0)
ss_est = generate_signal(go_pwr=0, ss_pwr=1, fs_pwr=0)
fs_est = generate_signal(go_pwr=0, ss_pwr=0, fs_pwr=1)
Y_est = go_est + ss_est + fs_est
tot_err = np.sum(np.abs(Y - Y_est))
df = pd.DataFrame.from_dict({"Y": Y,
                             "go_estimate": go_est,
                             "successful_stop_estimate": ss_est,
                             "failed_stop_estimate": fs_est,
                             "Y_est": Y_est})
source = ColumnDataSource(df)


# Set up plot
thr = 0.5
plot = figure(plot_height=400, plot_width=800, title="Stop Signal Task",
              tools="crosshair,pan,reset,save,wheel_zoom",
              x_range=[0, duration], y_range=[np.min(Y)-thr, np.max(Y)+thr])
plot.xaxis.axis_label = "Scans"

for name, color in zip(df.columns, Dark2[5]):
    if "Y" in name:
        line_width = 4
        line_alpha = 0.6
    else:
        line_width = 2
        line_alpha = 1.
    plot.line(x='index',
              y=name,
              line_width=line_width,
              line_alpha=line_alpha,
              source=source,
              color=color,
              legend=value(name))

plot.legend.location = "top_left"
plot.legend.orientation = "horizontal"
plot.legend.click_policy = "hide"

tracker = Label(x=30, y=10, x_units='screen', y_units='screen',
                text='Total Error: {err}'.format(err=tot_err), render_mode='css',
                border_line_color='black', border_line_alpha=1.0,
                background_fill_color='white', background_fill_alpha=1.0)

plot.add_layout(tracker)

# Set up widgets
data_title = Div(text="<b>Y Options</b>", style={'font-size': '200%'}, width=200, height=30)
go_onset = Slider(title="go_onset", value=2.0, start=0.0, end=22.0, step=1.0)
ss_onset = Slider(title="successful_stop_onset", value=12.0, start=0.0, end=22.0, step=1.0)
fs_onset = Slider(title="failed_stop_onset", value=22.0, start=0.0, end=22.0, step=1.0)
go_pwr = Slider(title="go_pwr", value=1, start=0, end=5, step=0.5)
ss_pwr = Slider(title="successful_stop_pwr", value=2, start=0, end=5, step=0.5)
fs_pwr = Slider(title="failed_stop_pwr", value=3, start=0, end=5, step=0.5)
noise = Slider(title="noise", value=0, start=0, end=0.1, step=0.01)

model_title = Div(text="<b>Y_est Options</b>", style={'font-size': '200%'}, width=200, height=30)
go_beta_onset = Slider(title="go_beta_onset", value=2.0, start=0.0, end=22.0, step=1.0)
ss_beta_onset = Slider(title="successful_stop_beta_onset", value=12.0, start=0.0, end=22.0, step=1.0)
fs_beta_onset = Slider(title="failed_stop_beta_onset", value=22.0, start=0.0, end=22.0, step=1.0)
go_beta = Slider(title="go_beta", value=1, start=0, end=5, step=0.1)
ss_beta = Slider(title="successful_stop_beta", value=1, start=0, end=5, step=0.1)
fs_beta = Slider(title="failed_stop_beta", value=1, start=0, end=5, step=0.1)


data_widgets = [data_title,
                go_onset,
                ss_onset,
                fs_onset,
                go_pwr,
                ss_pwr,
                fs_pwr,
                noise]

est_widgets = [model_title,
               go_beta_onset,
               ss_beta_onset,
               fs_beta_onset,
               go_beta,
               ss_beta,
               fs_beta]


# final updates to plot
go_marker = Label(x=go_onset.value, text="Go Trial", y=90, y_units="screen",
                  x_offset=5, text_color=Dark2[5][1])
ss_marker = Label(x=ss_onset.value, text="SS Trial", y=90, y_units="screen",
                  x_offset=5, text_color=Dark2[5][2])
fs_marker = Label(x=fs_onset.value, text="FS Trial", y=90, y_units="screen",
                  x_offset=5, text_color=Dark2[5][3])
plot.add_layout(go_marker)
plot.add_layout(ss_marker)
plot.add_layout(fs_marker)

def update_data(attrname, old, new):
    # Generate the new curve
    Y = generate_signal(go_onset=go_onset.value,
                        ss_onset=ss_onset.value,
                        fs_onset=fs_onset.value,
                        go_pwr=go_pwr.value,
                        ss_pwr=ss_pwr.value,
                        fs_pwr=fs_pwr.value,
                        noise=noise.value)

    s = slice(0, len(source.data['Y']))
    source.patch({"Y": [(s, Y)]})
    tot_err = np.sum(np.abs(source.data['Y'] - source.data['Y_est']))
    tracker.text = 'Total Error: {err}'.format(err=tot_err)
    go_marker.x = go_onset.value
    ss_marker.x = ss_onset.value
    fs_marker.x = fs_onset.value


def update_est(attrname, old, new):
    go_est = generate_signal(go_onset=go_beta_onset.value,
                             go_pwr=go_beta.value,
                             ss_pwr=0, fs_pwr=0)
    ss_est = generate_signal(ss_onset=ss_beta_onset.value,
                             ss_pwr=ss_beta.value,
                             go_pwr=0, fs_pwr=0)
    fs_est = generate_signal(fs_onset=fs_beta_onset.value,
                             fs_pwr=fs_beta.value,
                             go_pwr=0, ss_pwr=0)
    Y_est = go_est + ss_est + fs_est

    s = slice(0, len(source.data['Y']))

    source.patch({"go_estimate": [(s, go_est)],
                  "successful_stop_estimate": [(s, ss_est)],
                  "failed_stop_estimate": [(s, fs_est)],
                  "Y_est": [(s, Y_est)]})
    tot_err = np.sum(np.abs(source.data['Y'] - source.data['Y_est']))
    tracker.text = 'Total Error: {err}'.format(err=tot_err)


for w in data_widgets[1:]:
    w.on_change('value', update_data)

for w in est_widgets[1:]:
    w.on_change('value', update_est)


# Set up layouts and add to document
all_widgets = data_widgets + est_widgets

data_inputs = column(data_widgets)
est_inputs = column(est_widgets)

curdoc().add_root(grid([plot, [data_inputs, est_inputs]]))
curdoc().title = "Stop Signal Task"
