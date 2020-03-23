# -*- coding: utf-8 -*-
import math
import os.path as osp
from pathlib import Path
import requests
import pandas as pd
import numpy as np
import datetime
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from flask_caching import Cache
from flask import request
import argparse
import plotly.graph_objects as go
import logging
logger = logging.getLogger(__name__)

external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
cache = Cache(app.server, config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': '/tmp/covid-tracking'
})

##################################
# Constants
##################################

PKG_DIR = Path(osp.abspath(__file__)).parent
DATA_DIR = PKG_DIR.parent / 'data'
TIMEOUT = 10 # Short timeout (10 seconds) to avoid excessive API traffic
BASE_URL = 'https://covidtracking.com/api/'

STATES_DATA = pd.read_csv(DATA_DIR / 'states.csv')
STATES = {
    r['code']: r.rename(index=lambda i: 'display' if i == 'name' else i).to_dict()
    for _, r in STATES_DATA.iterrows()
}
STATES['all'] = dict(id='all', display='All')
ALL_STATES_ID = 'all'
RS_LOGO = "https://images.plot.ly/logo/new-branding/plotly-logomark.png"

STATS = {
    r.id: dict(id=r.id, display=r.name)
    for r in pd.read_csv(DATA_DIR / 'metrics.csv').itertuples()
}
DEFAULT_STATE = ALL_STATES_ID
DEFAULT_METRIC = 'total'

##################################
# Page Layout and Callbacks
##################################

@cache.memoize(timeout=TIMEOUT)
def get_endpoint_data(endpoint):
    url = BASE_URL + endpoint
    return requests.get(url).json()

def add_state_metadata(df):
    if 'state' not in df:
        return df
    cols = ['state', 'name', 'centroid_lat', 'centroid_lon', 'population']
    df_states = STATES_DATA.rename(columns={'code': 'state'})[cols]
    df = pd.merge(df, df_states, on='state', how='left')
    return df

def add_percapita_stats(df):
    if 'population' not in df:
        return df
    for c in STATS.keys():
        if c in df:
            df[c + '_pc'] = 1E6 * df[c] / df['population']
    return df

def get_data(endpoint):
    df = pd.DataFrame(get_endpoint_data(endpoint))
    df = add_state_metadata(df)
    df = add_percapita_stats(df)
    return df

class API(object):

    @property
    def states_daily(self):
        return get_data('states/daily')

    @property
    def states_current(self):
        return get_data('states')

    @property
    def nation_daily(self):
        return get_data('us/daily')

api = API()

app.layout = html.Div([
    dbc.Navbar(
        [
            html.A(
                dbc.Row(
                    [
                        dbc.Col(html.Img(src=RS_LOGO, height="30px")),
                        dbc.Col(dbc.NavbarBrand("COVID Tracking Project", className="ml-2")),
                    ],
                    align="center",
                    no_gutters=True,
                ),
                href="#",
            ),
            dbc.NavbarToggler(id="navbar-toggler")
        ],
        color="dark",
        dark=True,
    ),
    html.Div(
        [

            ##################################
            # Header and Introduction
            ##################################
            dbc.Row(dbc.Col([
                    html.Div(
                        'U.S. COVID-19 Case Statistic Tracking',
                        style={'font-size': '36px'}
                    ),
                    html.Div([
                        html.Div('By Eric Czech',
                        style={'font-style': 'italic', 'display': 'inline'}),
                        html.Div('Data updated daily at 4 PM EST',
                        style={'font-style': 'italic', 'color': 'red', 'display': 'inline', 'paddingLeft': '10px'}),
                    ]),
                    html.Hr(),
                    dcc.Markdown(
                        'This tracker contains visualizations of data from the [COVID Tracking Project](https://covidtracking.com/).  This information '
                        'is updated daily and combines several efforts to aggregate and report curated Coronavirus testing data.  See '
                        '[here](https://covidtracking.com/about-tracker/) for more details on how this data is collected.  The '
                        '[API](https://covidtracking.com/api/) documentation may also be helpful for those looking for direct access to the same '
                        'information shown below.\n\n'
                        'Note that for all graphics, ```Total Tests Administered``` = ```Confirmed Positive Tests``` + ```Confirmed Negative Tests``` + ```Pending Tests``` (```Deaths``` are tracked separately).'
                    ),
                    html.H4('Case Statistics', style={'font-style': 'bold'}),
                    html.Div(dcc.Markdown(
                            'This figure shows selected case statistics across the entire U.S. by default, or for a '
                            'specific state if one is chosen in the dropdown.  Note that individual statistics can be '
                            'selected while hiding all others by **double-clicking on items in the legend on the right**.',
                        ),
                        style={'font-size': '12px', 'color': 'grey'}
                    ),
                ],
                width=6
            ), justify='center'),

            ##################################
            # First Visualization (statistics)
            ##################################

            dbc.Row([
                dbc.Col([
                        dcc.Dropdown(
                            id='stats-state',
                            options=[
                                {'label': v['display'], 'value': k}
                                for k, v in STATES.items()
                            ],
                            value=None,
                            style={'paddingLeft': '5px', 'margin-top': '10px'},
                            clearable=False,
                            placeholder='Choose State ...'
                        )
                    ],
                    width={'size': 2, 'offset': 8}
                )
            ]),
            dbc.Row([
                dbc.Col(html.Div(dcc.Graph(
                    id='plot-stats',
                    figure=go.Figure(data=[], layout=go.Layout(
                        plot_bgcolor='white', paper_bgcolor='white',
                        xaxis=go.layout.XAxis(showticklabels=False),
                        yaxis=go.layout.YAxis(showticklabels=False)
                    ))
                )), width={'size': 8, 'offset': 2})
            ]),
            dbc.Row(dbc.Col([
                    html.Hr(),
                    html.H4('State-Level Breakdowns', style={'font-style': 'bold'}),
                    html.Div(dcc.Markdown(
                            'This figure shows a single statistic across all U.S. states, with the map (left) containing '
                            'aggregate counts to date and the time series (right) containing daily observations.  Note '
                            'that individual states can be selected in the plot on the right while hiding all others by '
                            '**double-clicking on the corresponding item in the legend**.',
                        ),
                        style={'font-size': '12px', 'color': 'grey'}
                    )
                ],
                width=6),
                justify='center'
            ),

            ##################################
            # Second Visualization (states)
            ##################################
            dbc.Row([
                dbc.Col([
                        dcc.Dropdown(
                            id='states-metric',
                            options=[
                                {'label': v['display'], 'value': k}
                                for k, v in STATS.items()
                            ],
                            value=None,
                            style={'paddingLeft': '5px'},
                            clearable=False,
                            placeholder='Choose Statistic ...'
                        )
                    ],
                    width={'size': 2, 'offset': 9},
                    style={'padding-right': '2px'}
                ),
                dbc.Col([
                    dcc.Checklist(
                        id='states-options',
                        options=[
                            {'label': 'Per-capita', 'value': 'percapita'},
                            {'label': 'Logarithmic', 'value': 'logarithmic'},
                            {'label': 'Choropleth', 'value': 'choropleth'},
                        ],
                        value=['percapita', 'choropleth'],
                        style={'position': 'relative', 'margin-top': '-7px', 'margin-bottom': '-3px'},
                        labelStyle={
                            'font-family': 'arial', 'margin': '0px', 'border': '0px',
                            'padding': '0px', 'font-size': '12px', 'display': 'block'
                        }
                    )
                    ],
                    width={'size': 1},
                    style={'padding-left': '2px', 'margin-top': '0px'}
                )
            ]),
            dbc.Row([
                dbc.Col(html.Div(dcc.Graph(
                    id='map-states',
                    figure=go.Figure(data=[], layout=go.Layout(
                        plot_bgcolor='white', paper_bgcolor='white',
                        xaxis=go.layout.XAxis(showticklabels=False),
                        yaxis=go.layout.YAxis(showticklabels=False)
                    ))
                )), width=5),
                dbc.Col(html.Div(dcc.Graph(
                    id='plot-states',
                    figure=go.Figure(data=[], layout=go.Layout(
                        plot_bgcolor='white', paper_bgcolor='white',
                        xaxis=go.layout.XAxis(showticklabels=False),
                        yaxis=go.layout.YAxis(showticklabels=False)
                    ))
                )), width=7)
            ])
        ],
        style={'margin': 'auto', 'width': '98%'}
    ),

    dbc.CardFooter([
            html.Img(alt="Creative Commons License", src="https://i.creativecommons.org/l/by/4.0/88x31.png", style={'display': 'inline'}),
            html.Div('This work is licensed under a', style={'display': 'inline', 'margin-left': '5px', 'margin-right': '5px'}),
            html.A('Creative Commons Attribution 4.0 International License', rel="license", href="http://creativecommons.org/licenses/by/4.0/")
#<img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br />.
#This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>
    ]),

])

def log_action(action, *args, **kwargs):
    # Dump out info for post-hoc traffic monitoring
    app.logger.warning(f'ACTION|{request.remote_addr}|{datetime.datetime.now()}|{action}|{args}|{kwargs}')

@app.callback(Output('plot-stats', 'figure'), [
    Input('stats-state', 'value')
])
def update_stats_plot(state):
    log_action('update_stats_plot', state)
    if state is None:
        state = DEFAULT_STATE
    if state == ALL_STATES_ID:
        df = api.nation_daily
    else:
        df = api.states_daily
        df = df[df['state'] == state]
        if len(df) == 0:
            logger.warning(f'No data found for state {state}')
            return go.Figure()
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    data = [
        go.Scatter(x=df['date'], y=df[c], name=STATS[c]['display'])
        for c in sorted(STATS.keys())
        if c in df
    ]
    layout = go.Layout()
    fig = go.Figure(data=data, layout=layout)
    location = 'Nationwide' if state == ALL_STATES_ID else STATES[state]['display']
    title = f'COVID-19 Status ({location})'
    fig.update_layout(
        plot_bgcolor='white', paper_bgcolor='white',
        yaxis_type='linear', margin=dict(t=30, b=0),
        height=300, title=title
    )
    return fig


@app.callback(Output('map-states', 'figure'), [
    Input('states-metric', 'value'),
    Input('states-options', 'value')
])
def update_states_map(metric, options):
    log_action('update_states_map', metric, options)
    if metric is None:
        metric = DEFAULT_METRIC
    if 'percapita' in options:
        metric = metric + '_pc'

    df = api.states_current
    df = df[df[metric] > 0]

    if 'choropleth' in options:
        data = [
            go.Choropleth(
                locations=df['state'],
                z=df[metric],
                locationmode='USA-states',
                colorscale=[[0,'green'], [.1, 'yellow'], [1,'red']],
                colorbar_title='',
                text=df.apply(lambda r: f'{r["name"]}: {math.ceil(r[metric]):.0f}', axis=1),
                hoverinfo='text'
            )
        ]
    else:
        data = [
            go.Scattergeo(
                lat=df['centroid_lat'],
                lon=df['centroid_lon'],
                marker=dict(
                    size=(10 + 30 * (df[metric] / df[metric].max())).fillna(0),
                    color=df[metric],
                    cmin=0,
                    cmax=df[metric].max(),
                    colorscale=[[0, 'green'], [.1, 'yellow'], [1, 'red']],
                    colorbar_title=''
                ),
                text=df.apply(lambda r: f'{r["name"]}: {math.ceil(r[metric]):.0f}', axis=1),
                hoverinfo='text',
                locationmode='USA-states'
            )
        ]

    title = f'{STATS[metric.split("_")[0]]["display"]}{" per 100k Residents" if "percapita" in options else ""} (To Date)'
    layout = go.Layout(
        title=title,
        plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(r=0, l=0),
        geo=go.layout.Geo(scope='usa')
    )
    fig = go.Figure(data=data, layout=layout)
    return fig

@app.callback(Output('plot-states', 'figure'), [
    Input('states-metric', 'value'),
    Input('states-options', 'value')
])
def update_states_plot(metric, options):
    log_action('update_states_plot', options)
    if metric is None:
        metric = DEFAULT_METRIC
    if 'percapita' in options:
        metric = metric + '_pc'
    df = api.states_daily
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df = df.pivot_table(index='date', columns='state', values=metric)
    data = [
        go.Scatter(x=df[c].index, y=df[c], name=c)
        for c in df
    ]
    layout = go.Layout()
    fig = go.Figure(data=data, layout=layout)
    title = f'{STATS[metric.split("_")[0]]["display"]}{" per 100k Residents" if "percapita" in options else ""} (Daily)'
    fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', title=title, yaxis_type='log' if 'logarithmic' in options else 'linear')
    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run COVID dashboard.')
    parser.add_argument('--port', type=int, help='Server port', default=8050)
    parser.add_argument('--host', type=str, help='Server host', default='0.0.0.0')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    args = parser.parse_args()
    app.run_server(debug=args.debug, host=args.host, port=args.port)
