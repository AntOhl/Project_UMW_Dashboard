#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Dashboard
import plotly.express as px
import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import dash_daq as daq

# Classicals
import pandas as pd
import numpy as np

# from PIL import Image
import matplotlib.pyplot as plt

from os import getcwd, path
data_path = path.join(getcwd(), 'data')


#Python scripts
from functions import *
#from data import *

# to join a file with a path : path.join(data_path, 'name.csv')

# Running the dashboard

external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


# Layout
tab1 = html.Div(
    dbc.Row(
        [
            dbc.Col(
                [
                    dbc.Row(
                        html.Div(
                            'Forecasts',
                            className = 'chart_tab1_title'
                        )
                    ),
                    dbc.Row(
                        html.Div(id='space_1_1', style={'padding': 10})
                        ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                html.Label('Country'),
                                dcc.Dropdown(
                                    id='country',
                                    options=[{'label': i, 'value': i} for i in l_countries],
                                    multi=False,
                                    value='France'
                                )
                             ],
                                width={"size": 5, "offset": 1}
                            ),
                            dbc.Col(
                                [
                                html.Label('Movie'),
                                dcc.Dropdown(
                                    id='movie',
                                    value='Mirages'
                                )
                             ],
                             width={"size": 5}
                            ),
                        ]
                    ),
                    dbc.Row(
                        html.Div(id='space_1_1_2', style={'padding': 10})
                        ),
                    dcc.Graph(
                        id='forecast',
                        figure={ 'data': [], 'layout': go.Layout(plot_bgcolor= 'rgba(0, 0, 0, 0)', 
                            paper_bgcolor= 'rgba(0, 0, 0, 0)', xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False))}
                    )
                ],
                md=6
            ),
            dbc.Col(
                [
                    dbc.Row(
                        html.Div(
                            'Comparision',
                            className = 'chart_tab1_title'
                        )
                    ),
                    dbc.Row(
                        html.Div(id='space_1_2', style={'padding': 10})
                        ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                html.Label('Compared with'),
                                dcc.RadioItems(
                                    options=[
                                        {'label': '  other country', 'value': 'country_compare'},
                                        {'label': '  other movie', 'value': 'movie_compare'}],
                                    value='movie_compare',
                                    id='id_comparision'
                                )  
                             ],
                                md=6
                            ),
                            dbc.Col(
                                [
                                html.Label('Selection'),
                                dcc.Dropdown(
                                    id='value_compare',
                                    value='Vendetta'
                                )
                             ],
                                md=6
                            ),
                        ]
                    ),
                    dbc.Row(
                        html.Div(id='space_1_2_2', style={'padding': 10})
                        ),
                    dcc.Graph(
                        id='forecast_2',
                        figure={ 'data': [], 'layout': go.Layout(plot_bgcolor= 'rgba(0, 0, 0, 0)', 
                            paper_bgcolor= 'rgba(0, 0, 0, 0)', xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False))}
                    )
                ],
                md=6
            )
        ]
    )
)

# Layout
tab2 = html.Div(
    dbc.Row(
        [
            dbc.Col(
                [
                    dbc.Row(
                        html.Div(
                            'Market Selection',
                            className = 'chart_tab1_title'
                        )
                    ),
                    dbc.Row(
                        html.Div(id='space_2_1', style={'padding': 10})
                        ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                html.Label('Home Country'),
                                dcc.Dropdown(
                                    id='home_country',
                                    options=[{'label': i, 'value': i} for i in l_origin_countries],
                                    multi=False,
                                    value='FR'
                                )
                             ],
                                width={"size": 3, "offset": 1}
                            ),
                            dbc.Col(
                                [
                                html.Label('Market Country'),
                                dcc.Dropdown(
                                    id='market_country',
                                    value='United States'
                                )
                             ],
                                md=4
                            ),
                            dbc.Col(
                                [
                                html.Label('Forecasting period'),
                                dcc.RadioItems(
                                    options=[
                                        {'label': '  3 months', 'value': 3},
                                        {'label': '  6 months', 'value': 6}                                    ],
                                    value=6,
                                    id='horizen'
                                ),
                                html.Div(id='intermediate-value-query', style={'display': 'none'}),
                                html.Div(id='intermediate-value-tags', style={'display': 'none'})
                             ],
                                md=4
                            )
                        ]
                    ),
                    dbc.Row(
                        html.Div(id='space_2_1_2', style={'padding': 10})
                        ),
                    dcc.Graph(
                        id='tag_analysis_1',
                        figure={ 'data': [], 'layout': go.Layout(plot_bgcolor= 'rgba(0, 0, 0, 0)', 
                            paper_bgcolor= 'rgba(0, 0, 0, 0)', xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False))}
                    )
                ],
                md=6
            ),
            dbc.Col(
                [
                    dbc.Row(
                        html.Div(
                            'Revenue Prediction with Tags',
                            className = 'chart_tab1_title'
                        )
                    ),
                    dbc.Row(
                        html.Div(id='space_2_2_1', style={'padding': 10})
                        ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                html.Label('Tag Names'),
                                dcc.Dropdown(
                                    id='tag1',
                                    multi=False,
                                    placeholder='Tag name 1'
                                )
                             ],
                                md=3
                            ),
                            dbc.Col(
                                [
                                html.Label('Tag Values'),
                                dcc.Input(
                                    type='number',
                                    id='tag1_value',
                                    placeholder='Value',
                                    min=0, max=100,
                                    style={'width': 100}
                                )
                            ],
                                md=3
                            ),
                            dbc.Col(
                                [
                                html.Label('Tag Names'),
                                dcc.Dropdown(
                                    id='tag4',
                                    placeholder='Tag name 4'
                                )
                            ],
                                md=3
                            ),
                            dbc.Col(
                                [
                                html.Label('Tag Values'),
                                dcc.Input(
                                    type='number',
                                    id='tag4_value',
                                    placeholder='Value',
                                    min=0, max=100,
                                    style={'width': 100}
                                )
                             ],
                                md=3
                            )
                        ]
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                dcc.Dropdown(
                                    id='tag2',
                                    multi=False,
                                    placeholder='Tag name 2'
                                ),
                                md=3
                            ),
                            dbc.Col(
                                dcc.Input(
                                    type='number',
                                    id='tag2_value',
                                    placeholder='Value',
                                    min=0, max=100,
                                    style={'width': 100}
                                )
                            ),
                            dbc.Col(
                                dcc.Dropdown(
                                    id='tag5',
                                    multi=False,
                                    placeholder='Tag name 5'
                                ),
                                md=3
                            ),
                            dbc.Col(
                                dcc.Input(
                                    type='number',
                                    id='tag5_value',
                                    placeholder='Value',
                                    min=0, max=100,
                                    style={'width': 100}
                                ),
                                md=3
                            )
                        ]
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                dcc.Dropdown(
                                    id='tag3',
                                    multi=False,
                                    placeholder='Tag name 3'
                                ),
                                md=3
                            ),
                            dbc.Col(
                                dcc.Input(
                                    type='number',
                                    id='tag3_value',
                                    placeholder='Value',
                                    min=0, max=100,
                                    style={'width': 100}
                                ),
                                md=3
                            ),
                            dbc.Col(
                                dcc.Dropdown(
                                    id='tag6',
                                    multi=False,
                                    placeholder='Tag name 6'
                                ),
                                md=3
                            ),
                            dbc.Col(
                                dcc.Input(
                                    type='number',
                                    id='tag6_value',
                                    placeholder='Value',
                                    min=0, max=100,
                                    style={'width': 100}
                                ),
                                md=3
                            )
                        ]
                    ),
                    dbc.Row(
                        html.Div(id='space_3_1_2', style={'padding': 5})
                        ),
                    dbc.Row(
                        [
                        html.Div('The expected revenue with these weighted tags is :',
                            style={"font-size":'100%', "text-align":"left"})
                        ]
                        ),
                    dbc.Row(
                        [
                        html.Div(id='space_3_1_3', style={'padding': 10}),
                        html.Div(id='revenue', style={"font-family":"courier", "font-size":'160%', "text-align":"left"})
                        ]
                        ),
                    dbc.Row(
                        [
                        dbc.Col(
                            [
                            html.Div(style={'padding': 10, "text-align":"left"}),
                            html.Div('Most similar movies related to tags:' ,id='space_3_1_4', style={'padding': 15, "text-align":"left"}),
                            dcc.Dropdown(
                                        id='similar_movies',
                                        multi=False,
                                        placeholder='Select a movie'
                                    ),
                            html.Div('Revenue of this movie for the period:' ,id='space_3_1_2_4', style={'padding': 10, "text-align":"left"}),
                            html.Div(id='revenue_similar_movie', style={'padding': 5, "font-family":"courier", "text-align":"left"})
                            ],
                            md=4
                            ),
                        dbc.Col(
                            dcc.Graph(
                                id='tags_analysis_similar_movie',
                                figure={'data': [], 'layout': go.Layout(plot_bgcolor= 'rgba(0, 0, 0, 0)', 
                                    paper_bgcolor= 'rgba(0, 0, 0, 0)', xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                                    yaxis=dict(showticklabels=False, showgrid=False, zeroline=False))}
                                ),
                            md=8
                            )
                        ]
                        ),
                    dbc.Row(
                        html.Div(id='space_3_1_5', style={'padding': 10})
                        )
                ],
                md=6
            )
        ]
    )
)

tab3 = html.Div(
    dbc.Row(
        [
            dbc.Col(
                [
                    dbc.Row(
                        html.Div(
                            'Movie Recommendation',
                            className = 'chart_tab1_title'
                            )
                        ),
                    dbc.Row(
                        html.Div(id='space_3_1', style={'padding': 10})
                        ),
                    dbc.Row(
                        [
                        dbc.Col(
                            [
                            html.Label('Select type of market: '),
                            dcc.RadioItems(
                                id='association_platform',
                                options=[{'label': ' sources  ', 'value': 'sources'},
                                {'label': ' countries  ', 'value': 'countries'},
                                {'label': ' platforms  ', 'value': 'platforms'}],
                                value='sources')
                            ],
                            width={"size": 4, "offset": 1}
                            ),
                        dbc.Col(
                            [
                            html.Label('Query a Movie'),
                            dcc.Dropdown(
                                id='movie_query',
                                multi=False)
                            ],
                            md=7
                            )
                        ]
                        ),
                    dcc.Graph(
                        id='table',
                        figure={ 'data': [], 'layout': go.Layout(plot_bgcolor= 'rgba(0, 0, 0, 0)', 
                            paper_bgcolor= 'rgba(0, 0, 0, 0)', xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False))}
                    )
                ],
                md=6
            ),
            dbc.Col(
                [
                    dbc.Row(
                        html.Div(
                            'Most Similar Movies, The Milky Way',
                            className = 'chart_tab1_title'
                        )
                    ),
                    dbc.Row(
                        html.Div(id='space_3_2', style={'padding': 10})
                        ),
                    dbc.Row(
                        html.Div('Visualisation')
                        ),
                    dbc.Row(
                        dcc.RadioItems(
                            id='milky_arg',
                            options=[{'label': ' all movies  ', 'value': 'all_movies'},
                            {'label': ' queried movie  ', 'value': 'queried_movie'}],
                            value='all_movies')
                    ),
                    dbc.Row(
                        html.Div(id='space_3_2_2', style={'padding': 10})
                        ),
                    dcc.Graph(
                        id='milky_way',
                        figure={ 'data': [], 'layout': go.Layout(plot_bgcolor= 'rgba(0, 0, 0, 0)', 
                            paper_bgcolor= 'rgba(0, 0, 0, 0)', xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False))}
                    )
                ],
                md=6
            )
        ]
    )
)

tabs_styles = {
    'height': '44px'
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'backgroundColor': "#171b26"
}

tab_selected_style = {
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '6px'
}

app.layout = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    html.H6(
                        '🎬 Movie Performances 🎬',
                        className="banner"
                    ),
                    md=8
                ),
                dbc.Col(
                    html.Img(
                        src=app.get_asset_url(
                            "essec_logo.png"
                        ),
                        className="banner"
                    ),
                    md=1
                ),
                dbc.Col(
                    html.Img(
                        src=app.get_asset_url(
                            "centralesupelec_logo.png"
                        ),
                        className="banner"
                    ),
                    md=1
                ),
                dbc.Col(
                    html.Img(
                        src=app.get_asset_url(
                            "UMW_logo.png"
                        ),
                        className="banner"
                    ),
                    md=1
                )


            ]),
        dcc.Tabs(
            [
                dcc.Tab(
                    label='Forecaster',
                    children=tab1,
                    style=tab_style,
                    selected_style=tab_selected_style

                ),
                dcc.Tab(
                    label='Tag Analysis',
                    children=tab2,
                    style=tab_style,
                    selected_style=tab_selected_style
                ),
                dcc.Tab(
                    label='Recommendation System',
                    children=tab3,
                    style=tab_style,
                    selected_style=tab_selected_style
                )
            ]
        )
    ]
)




@app.callback(
    Output('movie', 'options'),
    [Input('country', 'value')])
def update_movie_list(country):
    # Declare a global variable    
    df_country = df_tab1[df_tab1.Country == country]
    df_movie = df_titles_to_id[df_titles_to_id['Movie_id'].isin(df_country['Movie_id'].unique())]
    return [{'label': i, 'value': i} for i in df_movie['Title'].to_list()]

@app.callback(
    Output('forecast', 'figure'),
    [Input('movie', 'value'),
    Input('country', 'value')])
def update_figure(movie, country):
    # Declare a global variable
    df_country = df_tab1[df_tab1.Country == country]
    return forecast_wrap(country, df_country, movie)

@app.callback(
    Output('value_compare', 'options'),
    [Input('id_comparision', 'value'),
    Input('country', 'value'),
    Input('movie', 'value')])
def update_comparision_list(comparision, country, movie):
    # Declare a global variable
    if 'country' in comparision:
        df_countries = df_tab1[['Country', 'Title']][df_tab1.Title == movie].reset_index(drop=True).drop_duplicates()
        new_l_country = df_countries.Country.to_list()
        new_l_country.remove(country)
        return [{'label': i, 'value': i} for i in new_l_country]
    else:
        df_country = df_tab1[df_tab1.Country == country]
        df_movie = df_titles_to_id[df_titles_to_id['Movie_id'].isin(df_country['Movie_id'].unique())]
        new_l_movies = df_movie['Title'].to_list().copy()
        new_l_movies.remove(movie)
        return [{'label': i, 'value': i} for i in new_l_movies]

@app.callback(
    Output('forecast_2', 'figure'),
    [Input('id_comparision', 'value'),
    Input('value_compare', 'value'),
    Input('movie', 'value'),
    Input('country', 'value')])
def update_figure_2(comparision, value_compare, movie, country):
    # Declare a global variable
    if 'country' in comparision:
        df_country = df_tab1[df_tab1.Country == value_compare]
        return forecast_wrap(value_compare, df_country, movie)
    else:
        df_country = df_tab1[df_tab1.Country == country]
        return forecast_wrap(country, df_country, value_compare)



### TAB 2

#LEFT PART
@app.callback(
    Output('market_country', 'options'),
    [Input('home_country', 'value')])
def update_market_country_list(home_country):
    # Declare a global variable    
    l = get_martket_country(df_tab2, home_country, limit=10)
    return [{'label': i, 'value': i} for i in l]

@app.callback(
    Output('tag_analysis_1', 'figure'),
    [Input('home_country', 'value'),
    Input('market_country', 'value'),
    Input('horizen', 'value')])
def update_figure_3(home_country, market_country, horizen):

    return tag_analysis_wrap(df_tab2, home_country, market_country, horizen)


#INTERMEDIATE VALUES
@app.callback(
    Output('intermediate-value-query', 'children'),
    [Input('home_country', 'value'),
    Input('market_country', 'value'),
    Input('horizen', 'value')])
def intermediate_tab2_query(home_country, market_country, horizen):
    # Declare a global variable    
    horizen = horizen * 30
    country_in = [home_country]
    country_out = [market_country]
    df_query, _ = pipeline(df_tab2, tags, country_in, country_out, horizen)
    return df_query.to_json(date_format='iso', orient='split')

@app.callback(
    Output('intermediate-value-tags', 'children'),
    [Input('home_country', 'value'),
    Input('market_country', 'value'),
    Input('horizen', 'value')])
def intermediate_tab2_tags(home_country, market_country, horizen):
    # Declare a global variable
    horizen = horizen * 30
    country_in = [home_country]
    country_out = [market_country] 
    _ , df_tags = pipeline(df_tab2, tags, country_in, country_out, horizen)
    return df_tags.to_json(date_format='iso', orient='split')


#TAGS CHOICES
@app.callback(
    Output('tag1', 'options'),
    [Input('intermediate-value-tags', 'children')])
def update_tag_list_tab_2_1(df_tags_json):
    df_tags = pd.read_json(df_tags_json, orient='split')
    data = df_tags[df_tags['Count']>10].sort_values('Median',ascending=False).set_index('Tag')
    return [{'label': i, 'value': i} for i in list(data.index)]
@app.callback(
    Output('tag2', 'options'),
    [Input('intermediate-value-tags', 'children')])
def update_tag_list_tab_2_2(df_tags_json):
    df_tags = pd.read_json(df_tags_json, orient='split')
    data = df_tags[df_tags['Count']>10].sort_values('Median',ascending=False).set_index('Tag')
    return [{'label': i, 'value': i} for i in list(data.index)]
@app.callback(
    Output('tag3', 'options'),
    [Input('intermediate-value-tags', 'children')])
def update_tag_list_tab_2_3(df_tags_json):
    df_tags = pd.read_json(df_tags_json, orient='split')
    data = df_tags[df_tags['Count']>10].sort_values('Median',ascending=False).set_index('Tag')
    return [{'label': i, 'value': i} for i in list(data.index)]
@app.callback(
    Output('tag4', 'options'),
    [Input('intermediate-value-tags', 'children')])
def update_tag_list_tab_2_4(df_tags_json):
    df_tags = pd.read_json(df_tags_json, orient='split')
    data = df_tags[df_tags['Count']>10].sort_values('Median',ascending=False).set_index('Tag')
    return [{'label': i, 'value': i} for i in list(data.index)]
@app.callback(
    Output('tag5', 'options'),
    [Input('intermediate-value-tags', 'children')])
def update_tag_list_tab_2_5(df_tags_json):
    df_tags = pd.read_json(df_tags_json, orient='split')
    data = df_tags[df_tags['Count']>10].sort_values('Median',ascending=False).set_index('Tag')
    return [{'label': i, 'value': i} for i in list(data.index)]
@app.callback(
    Output('tag6', 'options'),
    [Input('intermediate-value-tags', 'children')])
def update_tag_list_tab_2_6(df_tags_json):
    df_tags = pd.read_json(df_tags_json, orient='split')
    data = df_tags[df_tags['Count']>10].sort_values('Median',ascending=False).set_index('Tag')
    return [{'label': i, 'value': i} for i in list(data.index)]


#FIGURE AND VALUES RIGHT PART
@app.callback(
    Output('revenue', 'children'),
    [Input('intermediate-value-tags', 'children'),
    Input('tag1', 'value'),
    Input('tag1_value', 'value'),
    Input('tag2', 'value'),
    Input('tag2_value', 'value'),
    Input('tag3', 'value'),
    Input('tag3_value', 'value'),
    Input('tag4', 'value'),
    Input('tag4_value', 'value'),
    Input('tag5', 'value'),
    Input('tag5_value', 'value'),
    Input('tag6', 'value'),
    Input('tag6_value', 'value')])
def update_market_country_list(df_tags_json, 
    tag1, val1, tag2, val2, tag3, val3, 
    tag4, val4, tag5, val5, tag6, val6):

    df_tags = pd.read_json(df_tags_json, orient='split')
    
    dict_rev = dict()


    dict_rev[tag1]= val1
    dict_rev[tag2]= val2
    dict_rev[tag3]= val3
    dict_rev[tag4]= val4
    dict_rev[tag5]= val5
    dict_rev[tag6]= val6

    dict_rev_2 = {k: v for k, v in dict_rev.items() if (k is not None) or (v is not None)}

    sol = compute_revenue_from_tags(df_tags, v=dict_rev_2)
    print_sol = str(sol) + ' €'
    return print_sol

@app.callback(
    Output('similar_movies', 'options'),
    [Input('intermediate-value-query', 'children'),
    Input('tag1', 'value'),
    Input('tag2', 'value'),
    Input('tag3', 'value'),
    Input('tag4', 'value'),
    Input('tag5', 'value'),
    Input('tag6', 'value')])
def update_most_similar_movies(df_query_json, 
    tag1, tag2, tag3, 
    tag4, tag5, tag6):

    df_query = pd.read_json(df_query_json, orient='split')

    l_tags = [tag1, tag2, tag3, tag4, tag5, tag6]
    if l_tags == [None]*6:
        return []
    else:
        l_tags = [i for i in l_tags if i is not None]
        l = get_similar_movies(l_tags, df_query=df_query)
        return [{'label': i, 'value': i} for i in l]

@app.callback(
    Output('tags_analysis_similar_movie', 'figure'),
    [Input('intermediate-value-query', 'children'),
    Input('similar_movies', 'value'),
    Input('tag1', 'value'),
    Input('tag2', 'value'),
    Input('tag3', 'value'),
    Input('tag4', 'value'),
    Input('tag5', 'value'),
    Input('tag6', 'value')])
def update_fig_similar_movies(df_query_json, similar_movies,
    tag1, tag2, tag3, 
    tag4, tag5, tag6):

    df_query = pd.read_json(df_query_json, orient='split')

    l_tags = [tag1, tag2, tag3, tag4, tag5, tag6]
    if l_tags == [None]*6:
        fig = go.Figure()
        fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)'
        })
        return fig
    else:
        l_tags = [i for i in l_tags if i is not None]
        fig = get_tags_of_similar_movies(df_query, similar_movies, l_tags)
        return fig

@app.callback(
    Output('revenue_similar_movie', 'children'),
    [Input('intermediate-value-query', 'children'),
    Input('similar_movies', 'value')])
def update_rev_similar_movies(df_query_json, similar_movies):

    df_query = pd.read_json(df_query_json, orient='split')
    rev = df_query[df_query.Title == similar_movies].iloc[0,4]
    rev = str(round(rev, 1)) + ' €'

    return rev




### TAB 3

@app.callback(
    Output('movie_query', 'options'),
    [Input('association_platform', 'value')])
def update_movie_query(association_platform):
    # Declare a global variable 
    df = getUniqueMovies(association_platform)
    return [{'label': i, 'value': i} for i in df.Title]

@app.callback(
    Output('milky_way', 'figure'),
    [Input('movie_query', 'value'),
    Input('milky_arg', 'value')])
def update_milky_way(movie_query, milky_arg):
    # Declare a global variable 
    if milky_arg == 'all_movies':
        return display_emb(query='')
    else:
        return display_emb(query=movie_query)



@app.callback(
    Output('table', 'figure'),
    [Input('movie_query', 'value'),
    Input('association_platform', 'value')])
def update_recommendation(movie_query, association_platform):
    # Declare a global variable 
    if association_platform == 'sources':
        return rankRecosByTransactions_source(movie_query)
    elif association_platform == 'countries':
        return rankRecosByTransactions_country(movie_query)
    else:
        return rankRecosByTransactions_sourcecountry(movie_query)


if __name__ == '__main__':
    app.run_server(debug=False)
