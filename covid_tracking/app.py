# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from flask_caching import Cache
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import requests
import pandas as pd

external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
cache = Cache(app.server, config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': '/tmp/covid-tracking'
})

TIMEOUT = 600
BASE_URL = 'https://covidtracking.com/api/'

@cache.memoize(timeout=TIMEOUT)
def get_data(endpoint):
    url = BASE_URL + endpoint
    print(url)
    return pd.DataFrame(requests.get(url).json())


class API(object):

    @property
    def states_daily(self):
        return get_data('states/daily')

    @property
    def states_current(self):
        return get_data('states')

api = API()

app.layout = html.Div(children=[
    html.H1('COVID Tracking'),
    html.Div(
        [
            dbc.Row([
                dbc.Col([
                    dcc.Dropdown(
                        id='map-metric',
                        options=[
                            {'label': 'Total', 'value': 'total'},
                            {'label': 'Positive', 'value': 'positive'},
                            {'label': 'Negative', 'value': 'negative'},
                            {'label': 'Pending', 'value': 'pending'},
                            {'label': 'Deaths', 'value': 'death'},
                        ],
                        value='total',
                        style={'paddingLeft': '5px'}
                    )
                ], width=3)
            ]),
            dbc.Row([
                dbc.Col(html.Div(dcc.Graph(
                    id='map-graph',
                    figure=go.Figure(data=[], layout=go.Layout(
                        plot_bgcolor='white', paper_bgcolor='white',
                        xaxis=go.XAxis(showticklabels=False),
                        yaxis=go.YAxis(showticklabels=False)
                    ))
                )), width=5),
                dbc.Col(html.Div(dcc.Graph(
                    id='timeseries-graph',
                    figure=go.Figure(data=[], layout=go.Layout(plot_bgcolor='white', paper_bgcolor='white'))
                )), width=7)
            ])
        ]
    )

])


@app.callback(Output('map-graph', 'figure'), [Input('map-metric', 'value')])
def update_map(metric):
    if metric is None:
        return None
    df = api.states_current
    fig = px.choropleth(
        locations=df['state'], locationmode="USA-states",
        color=df[metric], scope="usa",
        color_continuous_scale=[(0, "green"), (.1, "yellow"), (1, "red")]
    )
    title = f'Current {metric.title()} Cases'
    fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', margin=dict(r=0, l=0), title=title)
    return fig

@app.callback(Output('timeseries-graph', 'figure'), [Input('map-metric', 'value')])
def update_timeseries(metric):
    print(metric)
    df = api.states_daily
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df = df.pivot_table(index='date', columns='state', values=metric)
    data = [
        go.Scatter(x=df[c].index, y=df[c], name=c)
        for c in df
    ]
    layout = go.Layout()
    fig = go.Figure(data=data, layout=layout)
    title = f'{metric.title()} Cases by Day'
    fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', title=title)
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
