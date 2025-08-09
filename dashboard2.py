# ==============================================================================
# 0. IMPORT LIBRARIES
# ==============================================================================
import dash
import os
from dotenv import load_dotenv
import dash_bootstrap_components as dbc
from dash import dcc, html, callback_context, no_update
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy.signal import butter, lfilter, welch, iirnotch
import base64
import io
from supabase import create_client, Client

# ==============================================================================
# 1. SUPABASE SETUP
# ==============================================================================
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
BUCKET_NAME = "eeg-files"

try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    print(f"Supabase connection error: {e}")
    print("Please fill in your Supabase URL and Key in the script.")
    supabase = None

# ==============================================================================
# 2. SIGNAL PROCESSING FUNCTIONS
# ==============================================================================
def notch_filter(data, notch_freq, q_factor, fs):
    b, a = iirnotch(notch_freq, q_factor, fs)
    y = lfilter(b, a, data)
    return y

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def calculate_average_band_percentages(data, fs, lowcut, highcut):
    bands = {'Delta': (lowcut, 4), 'Theta': (4, 8), 'Alpha': (8, 13), 'Beta': (13, highcut)}
    channel_percentages = []
    for channel in data.columns:
        freqs, psd = welch(data[channel], fs=fs, nperseg=1024)
        total_power = np.trapz(psd[(freqs >= lowcut) & (freqs <= highcut)], freqs[(freqs >= lowcut) & (freqs <= highcut)])
        if total_power == 0: continue
        channel_bands = {}
        for band, (fmin, fmax) in bands.items():
            band_power = np.trapz(psd[(freqs >= fmin) & (freqs <= fmax)], freqs[(freqs >= fmin) & (freqs <= fmax)])
            channel_bands[band] = (band_power / total_power) * 100
        channel_percentages.append(channel_bands)
    avg_percentages = pd.DataFrame(channel_percentages).mean().to_dict()
    return avg_percentages

# ==============================================================================
# 3. DASH APPLICATION LAYOUT
# ==============================================================================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP])
server = app.server

def create_graph_card(graph_id, button_id, title):
    return dbc.Card([
        dbc.CardHeader([
            html.H5(title, className="card-title d-inline-block"),
            dbc.Button(html.I(className="bi bi-arrows-fullscreen"), id=button_id, color="secondary", outline=True, size="sm", className="float-end")
        ]),
        dbc.CardBody(dcc.Graph(id=graph_id, style={'height': '50vh'}))
    ])

app.layout = dbc.Container(fluid=True, style={'backgroundColor': '#f8f9fa', 'padding': '20px'}, children=[
    dcc.Store(id='current-dataframe-store'),
    dcc.Store(id='trigger-list-refresh'),

    html.H1("Interactive EEG Signal Dashboard", className="text-center text-primary mb-4"),
    
    dbc.Row([
        dbc.Col(md=3, children=[
            dbc.Card([
                dbc.CardHeader(html.H4("File Management")),
                dbc.CardBody([
                    html.Label("1. Select Folder:", className="fw-bold"),
                    dcc.Dropdown(id='folder-selector-dropdown', placeholder="Select a folder first...", className="mb-3"),
                    html.Hr(),
                    
                    html.Div(id='upload-wrapper', title="Please select a folder first", children=[
                        dcc.Upload(
                            id='upload-eeg-data',
                            children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
                            style={'width': '100%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px 0', 'cursor': 'pointer'},
                            multiple=True,
                            disabled=True
                        )
                    ]),
                    
                    dcc.Loading(id="loading-upload", type="circle", children=html.Div(id='upload-status')),
                    
                    html.Label("2. Select EEG File to Analyze:", className="fw-bold"),
                    html.Div(id='file-dropdown-wrapper', title="Please select a folder first", children=[
                        dcc.Dropdown(id='file-selector-dropdown', placeholder="Select a file...", className="mb-3", disabled=True)
                    ]),
                ])
            ], className="mb-4"),

            dbc.Card(id='controls-card', style={'display': 'none'}, children=[
                dbc.CardHeader(html.H4("Analysis Controls")),
                dbc.CardBody([
                    html.Label("Select EEG Channel:", className="fw-bold"),
                    dcc.Dropdown(id='channel-dropdown', clearable=False, className="mb-3"),
                    html.Label("Select Time Window (seconds):", className="fw-bold mt-3"),
                    dcc.RangeSlider(id='time-window-slider', step=1, tooltip={"placement": "bottom", "always_visible": True}),
                    html.Hr(),
                    html.Div(dbc.Switch(id='view-switch', label="Separate Raw/Filtered View", value=False), style={'cursor': 'pointer'}, className="mt-3")
                ])
            ], className="mb-4"),
            
            dbc.Card(id='brainwave-card', style={'display': 'none'}, children=[
                dbc.CardHeader(html.H4("Average Brainwave Distribution")),
                dbc.CardBody(id='brainwave-card-body')
            ])
        ]),
        
        dbc.Col(md=9, id='graphs-column', style={'display': 'none'}, children=[
            html.Div(id='combined-view-div', children=[create_graph_card('eeg-time-series-plot', 'expand-timeseries-btn', 'Time Series Analysis (Combined)')]),
            html.Div(id='separate-view-div', style={'display': 'none'}, children=[
                dbc.Row(dbc.Col(create_graph_card('raw-time-series-plot', 'expand-raw-btn', 'Raw Signal')), className="mb-4"),
                dbc.Row(dbc.Col(create_graph_card('filtered-time-series-plot', 'expand-filtered-btn', 'Filtered Signal')))
            ]),
            dbc.Row([
                dbc.Col(md=6, children=[create_graph_card('eeg-psd-plot', 'expand-psd-btn', 'Power Spectral Density (PSD)')], className="mb-4"),
                dbc.Col(md=6, children=[create_graph_card('correlation-heatmap', 'expand-heatmap-btn', 'Channel Correlation Heatmap')], className="mb-4")
            ])
        ])
    ]),
    
    dbc.Modal([dbc.ModalHeader(id="modal-header"), dbc.ModalBody(dcc.Graph(id="modal-graph", style={'height': '75vh'}))], id="graph-modal", size="xl", is_open=False)
])


# ==============================================================================
# 4. DASH CALLBACKS
# ==============================================================================

@app.callback(
    [Output('upload-status', 'children'), Output('trigger-list-refresh', 'data')],
    [Input('upload-eeg-data', 'contents')],
    [State('upload-eeg-data', 'filename'), State('folder-selector-dropdown', 'value')],
    prevent_initial_call=True
)
def upload_file(list_of_contents, list_of_names, selected_folder):
    if list_of_contents is None or supabase is None:
        return no_update, no_update
    
    if selected_folder is None:
        return dbc.Alert("Please select a folder before uploading files.", color="warning", duration=4000), no_update

    alerts = []
    upload_count = 0
    for content, name in zip(list_of_contents, list_of_names):
        try:
            content_type, content_string = content.split(',')
            decoded = base64.b64decode(content_string)
            upload_path = f"{selected_folder}/{name}" if selected_folder != "root" else name
            
            supabase.storage.from_(BUCKET_NAME).upload(file=decoded, path=upload_path, file_options={"content-type": "text/csv", "x-upsert": "true"})
            alerts.append(dbc.Alert(f"File uploaded successfully!", color="success", duration=3000))
            upload_count += 1
        except Exception as e:
            alerts.append(dbc.Alert(f"Error uploading {name}: {e}", color="danger", duration=4000))
            
    return alerts, {'uploads': upload_count}

@app.callback(
    Output('folder-selector-dropdown', 'options'),
    [Input('trigger-list-refresh', 'data')]
)
def update_folder_list(_):
    if supabase is None: return []
    try:
        res = supabase.storage.from_(BUCKET_NAME).list()
        folders = [{'label': '(root)', 'value': 'root'}] + \
                  [{'label': item['name'], 'value': item['name']} for item in res if item.get('id') is None]
        return folders
    except Exception as e:
        print(f"Error fetching folder list: {e}")
        return [{'label': '(root)', 'value': 'root'}]

@app.callback(
    [Output('upload-eeg-data', 'disabled'),
     Output('file-selector-dropdown', 'disabled'),
     Output('upload-wrapper', 'title'),
     Output('file-dropdown-wrapper', 'title')],
    [Input('folder-selector-dropdown', 'value')]
)
def toggle_controls_based_on_folder(selected_folder):
    if selected_folder is not None:
        return False, False, "", ""
    else:
        return True, True, "Please select a folder first", "Please select a folder first"

@app.callback(
    Output('file-selector-dropdown', 'options'),
    [Input('folder-selector-dropdown', 'value'),
     Input('trigger-list-refresh', 'data')]
)
def update_file_list(selected_folder, _):
    if selected_folder is None or supabase is None:
        return []

    try:
        path = selected_folder if selected_folder != "root" else ''
        res = supabase.storage.from_(BUCKET_NAME).list(path=path)
        files = [{'label': item['name'], 'value': item['name']} for item in res if item.get('id') is not None and item['name'] != '.emptyFolderPlaceholder']
        return files
    except Exception as e:
        print(f"Error fetching file list for folder '{selected_folder}': {e}")
        return []


@app.callback(
    [Output('current-dataframe-store', 'data'),
     Output('controls-card', 'style'),
     Output('brainwave-card', 'style'),
     Output('graphs-column', 'style')],
    [Input('file-selector-dropdown', 'value')],
    [State('folder-selector-dropdown', 'value')],
    prevent_initial_call=True
)
def load_selected_file(selected_file, selected_folder):
    if not selected_file or supabase is None:
        hidden_style = {'display': 'none'}
        return None, hidden_style, hidden_style, hidden_style

    try:
        full_path = f"{selected_folder}/{selected_file}" if selected_folder != "root" else selected_file
        response = supabase.storage.from_(BUCKET_NAME).download(full_path)
        df = pd.read_csv(io.BytesIO(response))
        
        if 'timestamp_name' in df.columns:
            df = df.drop('timestamp_name', axis=1)
        df.dropna(axis=1, how='all', inplace=True)
        
        visible_style = {'display': 'block'}
        return df.to_json(date_format='iso', orient='split'), visible_style, visible_style, visible_style
    except Exception as e:
        print(f"Error loading file {full_path}: {e}")
        hidden_style = {'display': 'none'}
        return None, hidden_style, hidden_style, hidden_style

@app.callback(
    [Output('channel-dropdown', 'options'),
     Output('channel-dropdown', 'value'),
     Output('time-window-slider', 'max'),
     Output('time-window-slider', 'marks'),
     Output('time-window-slider', 'value'), # *** FIX: Added this Output ***
     Output('brainwave-card-body', 'children')],
    [Input('current-dataframe-store', 'data')]
)
def update_ui_for_new_data(jsonified_data):
    if jsonified_data is None:
        # Provide default values for all outputs when no data is loaded
        return [], None, 10, {0: '0', 10: '10'}, [0, 10], []

    df = pd.read_json(jsonified_data, orient='split')
    
    channel_names = df.columns
    channel_options = [{'label': ch, 'value': ch} for ch in channel_names]
    default_channel = channel_names[0] if not channel_names.empty else None

    n_samples = len(df)
    total_duration = n_samples / 250
    num_marks = 10
    mark_step = max(60, int(total_duration / num_marks))
    slider_marks = {i: str(i) for i in range(0, int(total_duration), mark_step)}
    
    # *** Calculate and return a default value for the time slider ***
    default_time_window = [0, min(20, total_duration)]

    df_notched = df.apply(lambda col: notch_filter(col, 50.0, 30, 250))
    df_filtered_once = df_notched.apply(lambda col: bandpass_filter(col, 1.5, 30.0, 250))
    df_filtered = df_filtered_once.apply(lambda col: bandpass_filter(col, 1.5, 30.0, 250))
    
    avg_band_power = calculate_average_band_percentages(df_filtered, 250, 1.5, 30.0)
    brainwave_divs = [
        html.Div([
            html.Label(f"{band}: {avg_band_power.get(band, 0):.1f}%", className="fw-bold"),
            dbc.Progress(value=avg_band_power.get(band, 0), color=color, style={'height': '20px'})
        ], className="mb-2") for band, color in zip(['Delta', 'Theta', 'Alpha', 'Beta'], ['blue', 'orange', 'green', 'red'])
    ]

    return channel_options, default_channel, total_duration, slider_marks, default_time_window, brainwave_divs

@app.callback(
    [Output('combined-view-div', 'style'), Output('separate-view-div', 'style')],
    [Input('view-switch', 'value')]
)
def toggle_view(switch_on):
    if switch_on: return {'display': 'none'}, {'display': 'block'}
    else: return {'display': 'block'}, {'display': 'none'}

def process_data_for_graphs(jsonified_data):
    if not jsonified_data: return None, None
    df_raw = pd.read_json(jsonified_data, orient='split')
    df_notched = df_raw.apply(lambda col: notch_filter(col, 50.0, 30, 250))
    df_filtered_once = df_notched.apply(lambda col: bandpass_filter(col, 1.5, 30.0, 250))
    df_filtered = df_filtered_once.apply(lambda col: bandpass_filter(col, 1.5, 30.0, 250))
    return df_raw, df_filtered

@app.callback(
    Output('eeg-time-series-plot', 'figure'),
    [Input('current-dataframe-store', 'data'), Input('channel-dropdown', 'value'), Input('time-window-slider', 'value')]
)
def update_combined_time_series(jsonified_data, selected_channel, time_range):
    if not all([jsonified_data, selected_channel, time_range]): return go.Figure()
    df_raw, df_filtered = process_data_for_graphs(jsonified_data)
    if df_raw is None: return go.Figure()
    time_vector = np.arange(len(df_raw)) / 250
    start_idx, end_idx = int(time_range[0] * 250), int(time_range[1] * 250)
    time_sliced = time_vector[start_idx:end_idx]
    raw_sliced = df_raw[selected_channel][start_idx:end_idx]
    filtered_sliced = df_filtered[selected_channel][start_idx:end_idx]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_sliced, y=raw_sliced, name='Raw', line=dict(color='red', width=1)))
    fig.add_trace(go.Scatter(x=time_sliced, y=filtered_sliced, name='Filtered', line=dict(color='blue', width=1.5)))
    fig.update_layout(title=f"Channel: {selected_channel}", xaxis_title="Time (seconds)", yaxis_title="Amplitude (µV)", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

@app.callback(
    Output('raw-time-series-plot', 'figure'),
    [Input('current-dataframe-store', 'data'), Input('channel-dropdown', 'value'), Input('time-window-slider', 'value')]
)
def update_raw_time_series(jsonified_data, selected_channel, time_range):
    if not all([jsonified_data, selected_channel, time_range]): return go.Figure()
    df_raw, _ = process_data_for_graphs(jsonified_data)
    if df_raw is None: return go.Figure()
    time_vector = np.arange(len(df_raw)) / 250
    start_idx, end_idx = int(time_range[0] * 250), int(time_range[1] * 250)
    time_sliced = time_vector[start_idx:end_idx]
    raw_sliced = df_raw[selected_channel][start_idx:end_idx]
    fig = go.Figure(data=[go.Scatter(x=time_sliced, y=raw_sliced, name='Raw', line=dict(color='red', width=1))])
    fig.update_layout(title=f"Channel: {selected_channel}", xaxis_title="Time (seconds)", yaxis_title="Amplitude (µV)")
    return fig

@app.callback(
    Output('filtered-time-series-plot', 'figure'),
    [Input('current-dataframe-store', 'data'), Input('channel-dropdown', 'value'), Input('time-window-slider', 'value')]
)
def update_filtered_time_series(jsonified_data, selected_channel, time_range):
    if not all([jsonified_data, selected_channel, time_range]): return go.Figure()
    _, df_filtered = process_data_for_graphs(jsonified_data)
    if df_filtered is None: return go.Figure()
    time_vector = np.arange(len(df_filtered)) / 250
    start_idx, end_idx = int(time_range[0] * 250), int(time_range[1] * 250)
    time_sliced = time_vector[start_idx:end_idx]
    filtered_sliced = df_filtered[selected_channel][start_idx:end_idx]
    fig = go.Figure(data=[go.Scatter(x=time_sliced, y=filtered_sliced, name='Filtered', line=dict(color='blue', width=1.5))])
    fig.update_layout(title=f"Channel: {selected_channel}", xaxis_title="Time (seconds)", yaxis_title="Amplitude (µV)")
    return fig

@app.callback(
    Output('eeg-psd-plot', 'figure'),
    [Input('current-dataframe-store', 'data'), Input('channel-dropdown', 'value')]
)
def update_psd_plot(jsonified_data, selected_channel):
    if not all([jsonified_data, selected_channel]): return go.Figure()
    _, df_filtered = process_data_for_graphs(jsonified_data)
    if df_filtered is None: return go.Figure()
    freqs, psd = welch(df_filtered[selected_channel], fs=250, nperseg=1024)
    fig = go.Figure(data=[go.Scatter(x=freqs, y=psd, fill='tozeroy', line=dict(color='teal'))])
    fig.update_layout(title=f"Channel: {selected_channel}", xaxis_title="Frequency (Hz)", yaxis_title="Power (µV²/Hz)", xaxis_type='log', yaxis_type='log')
    return fig

@app.callback(
    Output('correlation-heatmap', 'figure'),
    [Input('current-dataframe-store', 'data')]
)
def update_heatmap_plot(jsonified_data):
    if not jsonified_data: return go.Figure()
    _, df_filtered = process_data_for_graphs(jsonified_data)
    if df_filtered is None: return go.Figure()
    correlation_matrix = df_filtered.corr()
    fig = go.Figure(data=go.Heatmap(z=correlation_matrix.values, x=correlation_matrix.columns, y=correlation_matrix.index, colorscale='Viridis'))
    fig.update_layout(title='Correlation of All Channels', xaxis_nticks=10)
    return fig

@app.callback(
    [Output("graph-modal", "is_open"), Output("modal-header", "children"), Output("modal-graph", "figure")],
    [Input("expand-timeseries-btn", "n_clicks"), Input("expand-raw-btn", "n_clicks"), Input("expand-filtered-btn", "n_clicks"), Input("expand-psd-btn", "n_clicks"), Input("expand-heatmap-btn", "n_clicks")],
    [State("graph-modal", "is_open"), State('eeg-time-series-plot', 'figure'), State('raw-time-series-plot', 'figure'), State('filtered-time-series-plot', 'figure'), State('eeg-psd-plot', 'figure'), State('correlation-heatmap', 'figure')],
    prevent_initial_call=True
)
def toggle_modal(ts_clicks, raw_clicks, filt_clicks, psd_clicks, heatmap_clicks, is_open, ts_fig, raw_fig, filt_fig, psd_fig, heatmap_fig):
    ctx = callback_context
    if not ctx.triggered: return is_open, "", go.Figure()
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == "expand-timeseries-btn": header, fig = "Time Series (Full Screen)", go.Figure(ts_fig)
    elif button_id == "expand-raw-btn": header, fig = "Raw Signal (Full Screen)", go.Figure(raw_fig)
    elif button_id == "expand-filtered-btn": header, fig = "Filtered Signal (Full Screen)", go.Figure(filt_fig)
    elif button_id == "expand-psd-btn": header, fig = "Power Spectral Density (Full Screen)", go.Figure(psd_fig)
    elif button_id == "expand-heatmap-btn": header, fig = "Channel Correlation Heatmap (Full Screen)", go.Figure(heatmap_fig)
    else: return not is_open, "", go.Figure()
    return not is_open, header, fig.update_layout(title=None)

# ==============================================================================
# 5. RUN THE APPLICATION
# ==============================================================================
if __name__ == '__main__':
    app.run(debug=True)
