import datetime
import requests
import os  
import dash_bootstrap_components as dbc
from dash import Dash, dcc, html, Input, Output, State, no_update
import plotly.graph_objs as go

# ------------------------------------------------------------------------------
# 1) APP-Konfiguration
# ------------------------------------------------------------------------------
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True
)
app.title = "Looper Machine Dashboard"

# ------------------------------------------------------------------------------
# 2) API-Konfiguration
# ------------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
ENDPOINT_LIVE = "/data/get_next_row_with_predictions"
ENDPOINT_BATCH = "/data/get_predictions_for_timeframe"

def fetch_data_live():
    try:
        resp = requests.get(API_BASE_URL + ENDPOINT_LIVE)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        print(f"Error fetching live data: {e}")
    return None

def fetch_data_batch(start_str, end_str):
    try:
        params = {"start_time": start_str, "end_time": end_str}
        resp = requests.get(API_BASE_URL + ENDPOINT_BATCH, params=params)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        print(f"Error fetching batch data: {e}")
    return None

def convert_iso_to_str(dt_iso):
    try:
        dt = datetime.datetime.fromisoformat(dt_iso)
        return dt.strftime("%H:%M:%S")
    except:
        return dt_iso

# ------------------------------------------------------------------------------
# 3) Farben + Plot-Helferfunktionen
# ------------------------------------------------------------------------------
STATE_COLORS = {
    "Idle": "#9ca3af",
    "Production": "#22c55e",
    "Remove Bottom Skin": "#fbbf24",
    "Block Loading": "#ff6b6b",
    "Remove Side Skin": "#a78bfa",
    "Remove Top Skin": "#ec4899",
    "Gluing": "#0ea5e9",
}

def create_empty_dark_figure():
    """Gibt einen leeren dunklen Plot zurück (kein weißes Flackern)."""
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor="#1e293b",
        plot_bgcolor="#1e293b",
        xaxis=dict(gridcolor="#334155", zerolinecolor="#334155"),
        yaxis=dict(gridcolor="#334155", zerolinecolor="#334155"),
        font=dict(color="#e2e8f0"),
        margin=dict(t=20, l=40, r=20, b=40),
    )
    return fig

def create_segmented_plot(data_list, model_key="knn"):
    """
    Zeichnet 2 Kurven (HTB, Agg) mit Zustand-Farbwechseln.
    Beide zeigen State, jede Kurve nur ihren eigenen Wert.
    Hover-Mode 'closest'.
    """
    sorted_data = sorted(data_list, key=lambda x: x["_time"])
    if len(sorted_data) < 2:
        return create_empty_dark_figure()

    times  = [row["_time"] for row in sorted_data]
    states = [row["predictions"][model_key] for row in sorted_data]
    y_htb  = [row["HTBVIst"] for row in sorted_data]
    y_agg  = [row["AggHoeheIst"] for row in sorted_data]

    # HTB-Traces
    htb_traces = []
    start_idx  = 0
    for i in range(1, len(sorted_data)):
        if states[i] != states[i-1]:
            seg_x = times[start_idx:i]
            seg_y = y_htb[start_idx:i]
            seg_st = states[i-1]
            color  = STATE_COLORS.get(seg_st, "#ffffff")
            htb_traces.append(
                go.Scatter(
                    x=[convert_iso_to_str(t) for t in seg_x],
                    y=seg_y,
                    mode="lines+markers",
                    line=dict(color=color, width=1),
                    marker=dict(color=color, size=4),
                    hovertemplate=(
                        "Time: %{x}<br>"
                        "HTBVIst: %{y}<br>"
                        f"State: {seg_st}<extra></extra>"
                    ),
                    showlegend=False,
                )
            )
            start_idx = i
    # letzer Segment-Abschluss
    seg_x = times[start_idx:]
    seg_y = y_htb[start_idx:]
    seg_st = states[-1]
    color  = STATE_COLORS.get(seg_st, "#ffffff")
    htb_traces.append(
        go.Scatter(
            x=[convert_iso_to_str(t) for t in seg_x],
            y=seg_y,
            mode="lines+markers",
            line=dict(color=color, width=1),
            marker=dict(color=color, size=4),
            hovertemplate=(
                "Time: %{x}<br>"
                "HTBVIst: %{y}<br>"
                f"State: {seg_st}<extra></extra>"
            ),
            showlegend=False,
        )
    )

    # Agg-Traces
    agg_traces = []
    start_idx  = 0
    for i in range(1, len(sorted_data)):
        if states[i] != states[i-1]:
            seg_x = times[start_idx:i]
            seg_y = y_agg[start_idx:i]
            seg_st = states[i-1]
            color  = STATE_COLORS.get(seg_st, "#ffffff")
            agg_traces.append(
                go.Scatter(
                    x=[convert_iso_to_str(t) for t in seg_x],
                    y=seg_y,
                    mode="lines+markers",
                    line=dict(color=color, width=1, dash="dot"),
                    marker=dict(color=color, size=4),
                    hovertemplate=(
                        "Time: %{x}<br>"
                        "AggHoeheIst: %{y}<br>"
                        f"State: {seg_st}<extra></extra>"
                    ),
                    showlegend=False,
                )
            )
            start_idx = i
    # letzer Segment-Abschluss
    seg_x = times[start_idx:]
    seg_y = y_agg[start_idx:]
    seg_st = states[-1]
    color  = STATE_COLORS.get(seg_st, "#ffffff")
    agg_traces.append(
        go.Scatter(
            x=[convert_iso_to_str(t) for t in seg_x],
            y=seg_y,
            mode="lines+markers",
            line=dict(color=color, width=1, dash="dot"),
            marker=dict(color=color, size=4),
            hovertemplate=(
                "Time: %{x}<br>"
                "AggHoeheIst: %{y}<br>"
                f"State: {seg_st}<extra></extra>"
            ),
            showlegend=False,
        )
    )

    fig = go.Figure(data=htb_traces + agg_traces)
    fig.update_layout(
        paper_bgcolor="#1e293b",
        plot_bgcolor="#1e293b",
        font=dict(color="#e2e8f0"),
        margin=dict(t=20, l=40, r=20, b=40),
        hovermode="closest",
        xaxis=dict(gridcolor="#334155", zerolinecolor="#334155"),
        yaxis=dict(gridcolor="#334155", zerolinecolor="#334155"),
    )
    return fig

def create_pie_state_times(all_results, model_key):
    if len(all_results) < 2:
        return create_empty_dark_figure()
    fromiso = datetime.datetime.fromisoformat
    sorted_data = sorted(all_results, key=lambda x: x["_time"])
    durations   = {s: 0.0 for s in STATE_COLORS}
    total_sec   = 0.0

    for i in range(len(sorted_data)-1):
        r1, r2 = sorted_data[i], sorted_data[i+1]
        st     = r1["predictions"][model_key]
        try:
            t1,t2 = fromiso(r1["_time"]), fromiso(r2["_time"])
            delta = (t2 - t1).total_seconds()
            if delta>0:
                durations[st]+=delta
                total_sec+=delta
        except:
            pass

    labels, values = [], []
    for s,sec in durations.items():
        if sec>0:
            labels.append(s)
            values.append(sec)
    if not labels:
        return create_empty_dark_figure()

    fig = go.Figure(
        go.Pie(labels=labels, values=values, hole=0.4, textinfo="label+percent")
    )
    fig.update_layout(
        paper_bgcolor="#1e293b",
        plot_bgcolor="#1e293b",
        font=dict(color="#e2e8f0"),
        margin=dict(t=20,l=40,r=20,b=40),
    )
    return fig

def calc_oee(results, model_key):
    if len(results) < 2:
        return (0,0,0)
    fromiso     = datetime.datetime.fromisoformat
    sorted_data = sorted(results, key=lambda x: x["_time"])
    durations   = {s:0.0 for s in STATE_COLORS}
    total_sec   = 0.0
    for i in range(len(sorted_data)-1):
        r1, r2 = sorted_data[i], sorted_data[i+1]
        st     = r1["predictions"][model_key]
        try:
            t1, t2 = fromiso(r1["_time"]), fromiso(r2["_time"])
            delta  = (t2 - t1).total_seconds()
            if delta>0:
                durations[st]+=delta
                total_sec+=delta
        except:
            pass

    if total_sec<1:
        return (0,0,0)

    still = durations["Idle"]
    prod  = durations["Production"]
    ruest = sum(durations[s] for s in durations if s not in ["Idle","Production"])
    denom_v = total_sec - still
    if denom_v<=0:
        verf=0
    else:
        verf= prod/denom_v

    denom_l = prod+ruest
    if denom_l<=0:
        leist=0
    else:
        leist= prod/denom_l
    return (verf, leist, verf*leist)

def oee_div(verf, leist, oee):
    return html.Div([
        html.P(f"Availability: {verf*100:.1f}%"),
        html.P(f"Performance: {leist*100:.1f}%"),
        html.H4(f"OEE: {oee*100:.1f}%", style={"color":"#22c55e"})
    ], style={"color":"white"})

# ------------------------------------------------------------------------------
# 5) Interval + Stores
# ------------------------------------------------------------------------------
live_interval = dcc.Interval(
    id="live-interval",
    interval=2000,
    n_intervals=0,
    disabled=False
)

live_data_store  = dcc.Store(id="live-data-store",  data=[], storage_type="memory")
batch_data_store = dcc.Store(id="batch-data-store", data={}, storage_type="memory")

# ------------------------------------------------------------------------------
# 6) Navbar + Page Layouts
# ------------------------------------------------------------------------------
navbar = dbc.NavbarSimple(
    brand="Looper Machine Dashboard",
    color="#1e293b",
    dark=True,
    children=[
        dbc.NavItem(dbc.NavLink("Live Monitoring", href="/live")),
        dbc.NavItem(dbc.NavLink("Batch Analysis", href="/batch")),
    ]
)

def layout_live_page():
    return html.Div([
        html.H1("Looper Machine - Real-Time Monitoring", style={"color":"white"}),
        html.P(
            "Streams new data every 2 seconds. Data resets on app restart, but persists on refresh/switch.",
            style={"color":"#94a3b8"}
        ),

        html.Div(
            dcc.Graph(id="live-plot", config={"displayModeBar":False},
                      style={"height":"60vh"}),
            style={"backgroundColor":"#1e293b","padding":"1rem","borderRadius":"0.5rem",
                   "marginBottom":"1rem"}
        ),

        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H3("Random Forest", style={"color":"white"}),
                    html.Div(
                        id="random-forest-metrics",
                        style={
                            "backgroundColor":"#334155",
                            "padding":"1rem",
                            "borderRadius":"0.5rem",
                            "marginBottom":"1rem"
                        }
                    ),
                    dcc.Graph(
                        id="random-forest-barplot",
                        style={"height":"30vh"},
                        config={"displayModeBar":False}
                    )
                ], style={"backgroundColor":"#1e293b","padding":"1rem","borderRadius":"0.5rem"})
            ], width=6),
            dbc.Col([
                html.Div([
                    # Modell: Neural Network
                    html.H3("Neural Network", style={"color":"white"}),
                    html.Div(
                        id="neural-network-metrics",
                        style={
                            "backgroundColor":"#334155",
                            "padding":"1rem",
                            "borderRadius":"0.5rem",
                            "marginBottom":"1rem"
                        }
                    ),
                    dcc.Graph(
                        id="neural-network-barplot",
                        style={"height":"30vh"},
                        config={"displayModeBar":False}
                    )
                ], style={"backgroundColor":"#1e293b","padding":"1rem","borderRadius":"0.5rem"})
            ], width=6)
        ])
    ], style={"padding":"1rem","minHeight":"100vh","backgroundColor":"#0f172a"})

def layout_batch_page():
    return html.Div([
        html.H1("Looper Machine - Batch Analysis", style={"color":"white"}),
        html.P(
            "Analyze historical data within a custom timeframe. "
            "Date/time remains on refresh, resets on app restart.",
            style={"color":"#94a3b8"}
        ),

        # Filter-Abschnitt
        html.Div([
            dbc.Row([
                dbc.Col([
                    html.Label("Start Date", style={"color":"#94a3b8", "marginRight":"1rem"}),
                    dcc.DatePickerSingle(
                        id="start-date-picker",
                        display_format="YYYY-MM-DD",
                        style={"backgroundColor":"#334155","color":"white"},
                        persistence=True,
                        persistence_type="memory"
                    )
                ], width=6),
                dbc.Col([
                    html.Label("End Date", style={"color":"#94a3b8", "marginRight":"1rem"}),
                    dcc.DatePickerSingle(
                        id="end-date-picker",
                        display_format="YYYY-MM-DD",
                        style={"backgroundColor":"#334155","color":"white"},
                        persistence=True,
                        persistence_type="memory"
                    )
                ], width=6)
            ], className="mb-3"),

            dbc.Row([
                dbc.Col([
                    html.Label("Start Time", style={"color":"#94a3b8", "marginRight":"1rem"}),
                    dbc.Input(
                        id="start-time-picker",
                        type="time",
                        value="08:00",
                        style={"backgroundColor":"#334155","color":"white"},
                        persistence=True,
                        persistence_type="memory"
                    )
                ], width=6),
                dbc.Col([
                    html.Label("End Time", style={"color":"#94a3b8", "marginRight":"1rem"}),
                    dbc.Input(
                        id="end-time-picker",
                        type="time",
                        value="10:00",
                        style={"backgroundColor":"#334155","color":"white"},
                        persistence=True,
                        persistence_type="memory"
                    )
                ], width=6)
            ], className="mb-3"),

            dbc.Button("Load Batch Data", id="load-batch-button", color="success", n_clicks=0),
        ], style={"backgroundColor":"#1e293b","padding":"1rem","borderRadius":"0.5rem","marginBottom":"1rem"}),

        # Plots mit dcc.Loading
        dcc.Loading(
            type="default",
            color="#22c55e",
            style={"marginTop": "1rem"},
            children=[
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H3("Random Forest Plot", style={"color":"white"}),
                            dcc.Graph(
                                id="random-forest-batch-plot",
                                figure=create_empty_dark_figure(),
                                style={"height":"40vh"}
                            ),
                        ], style={"backgroundColor":"#1e293b","padding":"1rem","borderRadius":"0.5rem"})
                    ], width=6),
                    dbc.Col([
                        html.Div([
                            html.H3("Neural Network Plot", style={"color":"white"}),
                            dcc.Graph(
                                id="neural-network-batch-plot",
                                figure=create_empty_dark_figure(),
                                style={"height":"40vh"}
                            ),
                        ], style={"backgroundColor":"#1e293b","padding":"1rem","borderRadius":"0.5rem"})
                    ], width=6),
                ], className="mb-3"),

                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H3("State Times (Random Forest)", style={"color":"white"}),
                            dcc.Graph(
                                id="random-forest-batch-states-chart",
                                figure=create_empty_dark_figure(),
                                style={"height":"40vh"}
                            ),
                            html.H3("OEE (Random Forest)", style={"color":"white"}),
                            html.Div(
                                id="random-forest-batch-oee",
                                style={
                                    "backgroundColor":"#334155",
                                    "padding":"0.5rem",
                                    "borderRadius":"0.5rem"
                                }
                            )
                        ], style={"backgroundColor":"#1e293b","padding":"1rem","borderRadius":"0.5rem"})
                    ], width=6),
                    dbc.Col([
                        html.Div([
                            html.H3("State Times (Neural Network)", style={"color":"white"}),
                            dcc.Graph(
                                id="neural-network-batch-states-chart",
                                figure=create_empty_dark_figure(),
                                style={"height":"40vh"}
                            ),
                            html.H3("OEE (Neural Network)", style={"color":"white"}),
                            html.Div(
                                id="neural-network-batch-oee",
                                style={
                                    "backgroundColor":"#334155",
                                    "padding":"0.5rem",
                                    "borderRadius":"0.5rem"
                                }
                            )
                        ], style={"backgroundColor":"#1e293b","padding":"1rem","borderRadius":"0.5rem"})
                    ], width=6),
                ])
            ]
        )
    ], style={"padding":"1rem","minHeight":"100vh","backgroundColor":"#0f172a"})

# ------------------------------------------------------------------------------
# 7) Root Layout
# ------------------------------------------------------------------------------
app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    live_interval,
    live_data_store,
    batch_data_store,
    navbar,
    html.Div(id="page-content", style={"minHeight":"100vh","backgroundColor":"#0f172a"})
], style={"backgroundColor":"#0f172a"})

# ------------------------------------------------------------------------------
# 8) Routing
# ------------------------------------------------------------------------------
@app.callback(
    Output("page-content","children"),
    Input("url","pathname")
)
def render_page_content(path):
    if path in ["/", "/live"]:
        return layout_live_page()
    elif path == "/batch":
        return layout_batch_page()
    else:
        return html.H1("404 - Not Found", style={"color":"white"})

# ------------------------------------------------------------------------------
# 9) Live-Daten: Interval => Store
# ------------------------------------------------------------------------------
@app.callback(
    Output("live-data-store","data"),
    Input("live-interval","n_intervals"),
    State("live-data-store","data")
)
def update_live_data(n_int, current_data):
    new_item = fetch_data_live()
    if new_item:
        current_data.append(new_item)
    return current_data

# ------------------------------------------------------------------------------
# 10) Live-Plot
# ------------------------------------------------------------------------------
@app.callback(
    Output("live-plot","figure"),
    Input("live-data-store","data")
)
def plot_live_data(data):
    if not data:
        return create_empty_dark_figure()
    times = [convert_iso_to_str(d["data"]["_time"]) for d in data]
    agg_vals= [d["data"]["AggHoeheIst"] for d in data]
    htb_vals= [d["data"]["HTBVIst"] for d in data]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=times,
            y=agg_vals,
            name="AggHoeheIst",
            mode="lines+markers",
            line=dict(color="#34d399", width=2),
            marker=dict(color="#34d399", size=6),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=times,
            y=htb_vals,
            name="HTBVIst",
            mode="lines+markers",
            line=dict(color="#60a5fa", width=2),
            marker=dict(color="#60a5fa", size=6),
        )
    )
    fig.update_layout(
        paper_bgcolor="#1e293b",
        plot_bgcolor="#1e293b",
        font=dict(color="#e2e8f0"),
        margin=dict(t=20, l=40, r=20, b=40),
        xaxis=dict(gridcolor="#334155", zerolinecolor="#334155"),
        yaxis=dict(gridcolor="#334155", zerolinecolor="#334155"),
        hovermode="closest"
    )
    return fig

# ------------------------------------------------------------------------------
# 11) Live-Metriken
# ------------------------------------------------------------------------------
@app.callback(
    Output("random-forest-metrics","children"),
    Output("neural-network-metrics","children"),
    Output("random-forest-barplot","figure"),
    Output("neural-network-barplot","figure"),
    Input("live-data-store","data")
)
def update_live_metrics(data):
    if not data:
        return "N/A", "N/A", create_empty_dark_figure(), create_empty_dark_figure()

    # Neuester Datensatz
    latest = data[-1]

    # API liefert:
    #   latest["predictions"]["random_forest"]
    #   latest["predictions"]["neural_network"]
    rf_info = latest["predictions"]["random_forest"]
    nn_info = latest["predictions"]["neural_network"]

    rf_pred = rf_info["prediction"]
    rf_conf = rf_info["confidence_score"]
    rf_proba= rf_info["predict_proba"]

    nn_pred = nn_info["prediction"]
    nn_conf = nn_info["confidence_score"]
    nn_proba= nn_info["predict_proba"]

    rf_div = html.Div([
        html.P(f"Prediction: {rf_pred}"),
        html.P(f"Confidence: {rf_conf:.2f}")
    ], style={"color":"white"})

    nn_div = html.Div([
        html.P(f"Prediction: {nn_pred}"),
        html.P(f"Confidence: {nn_conf:.2f}")
    ], style={"color":"white"})

    def create_bar(probas, color):
        if not probas:
            return create_empty_dark_figure()
        fig = go.Figure(
            go.Bar(
                x=list(probas.keys()),
                y=list(probas.values()),
                marker=dict(color=color)
            )
        )
        fig.update_layout(
            paper_bgcolor="#1e293b",
            plot_bgcolor="#1e293b",
            font=dict(color="#e2e8f0"),
            margin=dict(t=20, l=40, r=20, b=40),
            xaxis=dict(gridcolor="#334155", zerolinecolor="#334155"),
            yaxis=dict(gridcolor="#334155", zerolinecolor="#334155"),
        )
        return fig

    rf_fig = create_bar(rf_proba, "#34d399")
    nn_fig = create_bar(nn_proba, "#60a5fa")

    return rf_div, nn_div, rf_fig, nn_fig

# ------------------------------------------------------------------------------
# 12) Batch-Daten laden
# ------------------------------------------------------------------------------
@app.callback(
    Output("batch-data-store","data"),
    Input("load-batch-button","n_clicks"),
    State("start-date-picker","date"),
    State("start-time-picker","value"),
    State("end-date-picker","date"),
    State("end-time-picker","value"),
    State("batch-data-store","data"),
    prevent_initial_call=True
)
def load_batch_data(n_clicks, start_d, start_t, end_d, end_t, current_data):
    if not (start_d and start_t and end_d and end_t):
        return current_data

    start_str = f"{start_d}T{start_t}:00"
    end_str   = f"{end_d}T{end_t}:00"

    if current_data and isinstance(current_data, dict):
        stored_range = current_data.get("query_range", {})
        if stored_range.get("start") == start_str and stored_range.get("end") == end_str:
            print("Time range unchanged -> no new fetch")
            return current_data

    new_data = fetch_data_batch(start_str, end_str)
    if new_data:
        new_data["query_range"] = {"start": start_str, "end": end_str}
        return new_data

    return current_data

# ------------------------------------------------------------------------------
# 13) Batch-Plots
# ------------------------------------------------------------------------------
@app.callback(
    Output("random-forest-batch-plot","figure"),
    Output("neural-network-batch-plot","figure"),
    Output("random-forest-batch-states-chart","figure"),
    Output("neural-network-batch-states-chart","figure"),
    Output("random-forest-batch-oee","children"),
    Output("neural-network-batch-oee","children"),
    Input("batch-data-store","data")
)
def update_batch_plots(batch_data):
    if not batch_data or not isinstance(batch_data, dict):
        return (
            create_empty_dark_figure(),
            create_empty_dark_figure(),
            create_empty_dark_figure(),
            create_empty_dark_figure(),
            html.Div("No data available", style={"color":"white"}),
            html.Div("No data available", style={"color":"white"})
        )

    results = batch_data.get("results", [])
    if not results:
        return (
            create_empty_dark_figure(),
            create_empty_dark_figure(),
            create_empty_dark_figure(),
            create_empty_dark_figure(),
            html.Div("No data for the chosen timeframe", style={"color":"white"}),
            html.Div("No data for the chosen timeframe", style={"color":"white"})
        )

    rf_fig = create_segmented_plot(results, "random_forest")
    nn_fig = create_segmented_plot(results, "neural_network")

    rf_pie = create_pie_state_times(results, "random_forest")
    nn_pie = create_pie_state_times(results, "neural_network")

    v1, l1, o1 = calc_oee(results, "random_forest")
    v2, l2, o2 = calc_oee(results, "neural_network")

    def make_oee_div(verf, leist, oee):
        return html.Div([
            html.P(f"Availability: {verf*100:.1f}%"),
            html.P(f"Performance: {leist*100:.1f}%"),
            html.H4(f"OEE: {oee*100:.1f}%", style={"color":"#22c55e"})
        ], style={"color":"white"})

    return (
        rf_fig,
        nn_fig,
        rf_pie,
        nn_pie,
        make_oee_div(v1, l1, o1),
        make_oee_div(v2, l2, o2)
    )

# ------------------------------------------------------------------------------
# 14) App Start
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050, debug=True)
