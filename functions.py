#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Dashboard
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import dash_daq as daq

import plotly.express as px
import plotly.graph_objects as go

import matplotlib.pyplot as plt

from pmdarima.arima import auto_arima
from os import getcwd, path
data_path = path.join(getcwd(), 'data')


#Data for tab_1
df_tab1 = pd.read_csv(path.join(data_path, 'forecasting_movies_countries.csv'))
l_countries = df_tab1.Country.unique()
df_titles_to_id = df_tab1[['Movie_id', 'Title']].reset_index(drop=True).drop_duplicates()


#Data for tab_2
df_tab2 = pd.read_csv(path.join(data_path, 'movie_revenue_new_by_day.csv'), parse_dates=['Date'])
tags = pd.read_csv(path.join(data_path, "tags_clean.csv"), index_col='id')
df_countries = pd.DataFrame(df_tab2.groupby(['Country_of_origin', 'Country_of_market'])['Mama_id'].count()).reset_index()
l_origin_countries = df_countries[df_countries.Mama_id > 10]['Country_of_origin'].unique()


#Data for tab_3
#tab_3_1
df_tab3_1 = pd.read_csv(path.join(data_path, 'movie_revenue_new_by_day_and_platform.csv')).dropna()
s_portfolio = df_tab3_1.groupby(['Source'])['Mama_id'].unique()
mama2title = df_tab3_1[['Mama_id', 'Title']].drop_duplicates().set_index('Mama_id')
title2mama = df_tab3_1[['Mama_id', 'Title']].drop_duplicates().set_index('Title')

df1_s = pd.read_excel(path.join(data_path, 'df1_s.xlsx'), index_col = 0)
df1_c = pd.read_excel(path.join(data_path, 'df1_c.xlsx'), index_col = 0)
df1_sc = pd.read_excel(path.join(data_path, 'df1_sc.xlsx'), index_col = 0)
df_s_cos = pd.read_csv(path.join(data_path, 'similarity_by_source.csv'), index_col = 0)
df_c_cos = pd.read_csv(path.join(data_path, 'similarity_by_country.csv'), index_col = 0)
df_sc_cos = pd.read_csv(path.join(data_path, 'similarity_by_sourcecountry.csv'), index_col = 0)
df_s_reco = pd.read_csv(path.join(data_path, 'associationmetrics_by_source.csv'), index_col=0)
df_c_reco = pd.read_csv(path.join(data_path, 'associationmetrics_by_country.csv'), index_col=0)
df_sc_reco = pd.read_csv(path.join(data_path, 'associationmetrics_by_sourcecountry.csv'), index_col=0)

# Portfolio construction
c_portfolio = df_tab3_1.groupby(['Country_of_market'])['Mama_id'].unique()
sc_portfolio = df_tab3_1.groupby(['source_country'])['Mama_id'].unique()
s_portfolio = df_tab3_1.groupby(['Source'])['Mama_id'].unique()

#tab_3_2
df_tab3_2 = pd.read_csv(path.join(data_path, 'movie_transactions_new_by_day_and_platform.csv')).dropna()
tags_3 = pd.read_csv(path.join(data_path, 'tags_clean.csv'),sep=',')
tags_3 = tags_3.rename(columns={"id": "Mama_id"})
movie2vec = np.load(path.join(data_path, 'movie2vec.npy'))
l_movie_tab3 = list(title2mama.index)


#########
### TAB 1
#########


def forecast_wrap(country, data = df_tab1, movie = 'Requiem for a Dream', h = 3):

    # Filter out the dataframe by movie id (Pandas)
    movie_id = df_titles_to_id[df_titles_to_id['Title']==movie].iloc[0,0]
    df = data[data['Movie_id'] == movie_id]
    print(df.shape)
    # Reset the index
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    df.index = pd.to_datetime(df.index)

    # Remove incomplete data from the current month
    end_of_last_month = pd.to_datetime('today') + pd.tseries.offsets.MonthEnd(-1)
    df = df[:end_of_last_month]

    # Make a dataframe for FP prophet
    df = df.resample('1M')['Royalties'].sum()
    df = df.reset_index()
    df.columns = ['ds', 'y']

    # Fit the model
    # Log transformation to avoid negative predictions
    # Train an auto arima model
    model = auto_arima(np.log(df['y'] + 1), suppress_warnings=True)
    
    # Reverse pre-processing helper
    F = lambda x:np.exp(x) - 1
    
    # Make predictions
    pred = model.predict(n_periods=h, return_conf_int=True)
    pred_yhat = F(pred[0])
    lower_c = F(pred[1][:,0])
    upper_c = F(pred[1][:,1])
    
    # Prediction dataframe
    pred_date = df['ds'][-h:] + pd.tseries.offsets.MonthEnd(h)
    forecast = pd.DataFrame({'ds':pred_date,
                             'y':pred_yhat})
    
    # Confidence interval dataframe
    conf = pd.DataFrame({'ds':forecast['ds'], 
                        'yhat_upper':lower_c, 
                        'yhat_lower':upper_c})

    # Put the forecast and factual data into the same dataframe
    df['type'] = 'past'

    forecast['type'] = 'forecast'
    df = pd.concat([df, forecast],axis=0)
    #---------------------

    # Round the number down to 2 digits after the decimal points
    df['y'] = np.round(df['y'], 2)

    # Plot the result
    m_past = df['type'] == 'past'
    m_fore = df['type'] == 'forecast'

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['ds'][m_past], y=df['y'][m_past], mode='lines', name='past'))
    fig.add_trace(go.Scatter(x=df['ds'][m_fore], y=df['y'][m_fore], mode='lines+markers', name='forecast'))
    fig.add_trace(go.Scatter(x=[df['ds'][m_past].iloc[-1], df['ds'][m_fore].iloc[0]], 
                            y=[df['y'][m_past].iloc[-1], df['y'][m_fore].iloc[0]], 
                            mode='lines', name='forecast', 
                            hoverinfo='skip', line_color='#ff7f0e', showlegend=False))

    # Confidence interval
    fig.add_trace(go.Scatter(x=conf['ds'], y=conf['yhat_lower'], 
                             mode='lines', name='possible range', 
                             fill=None, line_color='#ff7f0e', opacity=0.1))
    fig.add_trace(go.Scatter(x=conf['ds'], y=conf['yhat_upper'], 
                             mode='lines', name='possible range', 
                             fill='tonexty', line_color='#ff7f0e', opacity=0.1))

    #The line connecting past and forecasted points
    fig.update_layout(
        title='Royalties Forecasting for ' + movie + ' in ' + country ,
        xaxis_title='Year',
        yaxis_title='Royalities (in euros)',
        font=dict(
            family="Arial",
            color="#7f7f7f"
        ),
        xaxis=dict(
            showline=True,
            showgrid=True,
            showticklabels=True,
            linecolor='rgb(159, 166, 183)',
            linewidth=2,
            ticks='outside',
            tickfont=dict(
                family='Arial',
                size=16,
                color='rgb(159, 166, 183)',
            ),
        ),
        yaxis=dict(
            showgrid=True,
            zeroline=True,
            showline=False,
            showticklabels=True,
            linecolor='rgb(115,115,115)',
            tickfont=dict(
                family='Arial',
                size=16,
                color='rgb(159, 166, 183)',
            ),

        ),
        autosize=True,
        width=600,
        height=400,
        margin=dict(
            autoexpand=False,
            l=50,
            r=30,
            t=40,
            b=70        ),
        showlegend=False,
    )
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)'
    })
    fig.update_traces(
        marker_color= 'rgb(255, 86, 65)',
        marker_line_color='rgb(240,240,240)',
        marker_line_width=1.5,
        opacity=0.9
    )

    return fig






#########
### TAB 2
#########

def tag_analysis_wrap(df=df_tab2, country_in='FR', country_out='United States', horizen=6, tags=tags, min_nb_movie=10, top=20):


    country_in = [country_in]
    country_out = [country_out]

    df_query, df_tags = pipeline(df, tags, country_in, country_out, horizen)
    # Plot the the top 20 tags with at least 10 movies
    data = df_tags[df_tags['Count']>min_nb_movie].sort_values('Median',ascending=False)
    data = data.iloc[:top]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=data.Tag.to_list(), y=data.Median.to_list()))
    title = 'Top {} Tags'.format(top)
    fig.update_layout(title_text=title)
    fig.update_layout(
        yaxis_title='Median Royalities (in euros)',
        font=dict(
            family="Arial",
            color="#7f7f7f"
        ),
        xaxis=dict(
            showline=True,
            showgrid=True,
            showticklabels=True,
            linecolor='rgb(159, 166, 183)',
            linewidth=2,
            ticks='outside',
            tickangle=45,
            tickfont=dict(
                family='Arial',
                size=16,
                color='rgb(159, 166, 183)'
            ),
        ),
        yaxis=dict(
            showgrid=True,
            zeroline=True,
            showline=False,
            showticklabels=True,
            linecolor='rgb(115,115,115)',
            tickfont=dict(
                family='Arial',
                size=16,
                color='rgb(159, 166, 183)',
            ),

        ),
        autosize=True,
        width=600,
        margin=dict(
            autoexpand=False,
            l=60,
            r=30,
            t=40,
            b=120
        ),
        showlegend=False,
    )
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)'
    })
    fig.update_traces(
        marker_color= 'rgb(255, 86, 65)',
        marker_line_color='rgb(240,240,240)',
        marker_line_width=1.5,
        opacity=0.9
    )
    return fig


def rev_stat(tag_col, f, df):
    mask = tag_col>0
    rev = df.loc[mask,'Royalties']
    return f(rev)

def generate_df_tags(df):
    # Generate tag stats
    stats = {}
    stats['Count'] = df.iloc[:,5:].apply(rev_stat, args=[len, df])
    stats['Median'] = df.iloc[:,5:].apply(rev_stat, args=[np.median, df])
    stats['Mean'] = df.iloc[:,5:].apply(rev_stat, args=[np.mean, df])
    stats['Sum'] = df.iloc[:,5:].apply(rev_stat, args=[np.sum, df])
    stats['STD'] = df.iloc[:,5:].apply(rev_stat, args=[np.std, df])

    # Save to a df
    df_tags = pd.DataFrame(stats)
    df_tags.index.name = 'Tag'
    df_tags = df_tags.reset_index()
    
    return df_tags

def query_tag_list(df, tag):
    return df[(df[tag] > 0).any(axis=1)]

# Generate a dataframe for analysis
def pipeline(df, tags, country_in, country_out, horizen):
    
    # horizen = -1 means taking the whole horizen
    if horizen == -1:
        horizen = pd.Timedelta(99999, unit='D')
    else:
        horizen = pd.Timedelta(horizen * 30, unit='D') 
    
    # Filter by country
    mask = df['Country_of_origin'].isin(country_in) & df['Country_of_market'].isin(country_out)
    df = df[mask]
    
    # Filter by horizen
    df_temp = df.groupby(['Mama_id','Movie_id','Title','Country_of_origin', 'Country_of_market'])
    mask = df_temp['Date'].apply(lambda x:(x - min(x))<=horizen)
    df = df[mask]
    
    # Aggregate by country
    df = df.groupby(['Mama_id','Movie_id','Title','Country_of_origin', 'Country_of_market']).sum()
    df = df.reset_index().set_index('Mama_id')

    # Merge with tags
    df = df.merge(tags,left_index=True, right_index=True, how='left')
   
    # Generate a df for EDA
    df_tags = generate_df_tags(df)
    return df, df_tags


def get_martket_country(df, home_country, limit=10):
    df_countries = pd.DataFrame(df.groupby(['Country_of_origin', 'Country_of_market'])['Mama_id'].count()).reset_index()
    df_home = df_countries[df_countries.Country_of_origin == home_country]
    l = df_home[df_home.Mama_id > limit].Country_of_market.to_list()
    return l


# Output a revenue category by country_in, country_out, horizen and tag vectors
def compute_revenue_from_tags(df_tags, v = {'fantasy':30, 'false identities':70}, tags=tags):

    # Plot the the top 20 tags with at least 10 movies
    data = df_tags[df_tags['Count']>10].sort_values('Median',ascending=False).set_index('Tag')
    
    # Predict revenues using tag vectors
    pred = data.loc[v.keys()][['Median']]
    pred['weight'] = v.values()
    rev = (pred['Median'] * pred['weight']).sum()
    rev = (rev//100000) * 1000

    return rev


def get_similar_movies(l_tags, df_query):
    df = query_tag_list(df_query, l_tags).sort_values(l_tags, ascending=False)
    return df.Title


def get_tags_of_similar_movies(df_query, movie, l_tags):

    df = query_tag_list(df_query, l_tags).sort_values(l_tags, ascending=False)
    df_calc = df[df.Title == movie].iloc[:, 5:].T
    df_calc2 = df_calc[df_calc.iloc[:, 0] > 0].reset_index()
    df_calc2.columns = ['tags', 'value']
    df_calc2 = df_calc2.sort_values('value',ascending=False)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df_calc2.tags.to_list(), y=df_calc2.value.to_list()))
    fig.update_layout(
        font=dict(
            family="Arial",
            color="#7f7f7f"
        ),
        xaxis=dict(
            showline=True,
            showgrid=True,
            showticklabels=True,
            linecolor='rgb(159, 166, 183)',
            linewidth=2,
            ticks='outside',
            tickangle=45,
            tickfont=dict(
                family='Arial',
                size=16,
                color='rgb(159, 166, 183)'
            ),
        ),
        yaxis=dict(
            showgrid=True,
            zeroline=True,
            showline=False,
            showticklabels=True,
            linecolor='rgb(115,115,115)',
            tickfont=dict(
                family='Arial',
                size=16,
                color='rgb(159, 166, 183)',
            ),

        ),
        autosize=True,
        height=300,
        width=400,
        margin=dict(
            autoexpand=False,
            l=60,
            r=30,
            t=10,
            b=120
        ),
        showlegend=False,
    )
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)'
    })
    fig.update_traces(
        marker_color= 'rgb(255, 86, 65)',
        marker_line_color='rgb(240,240,240)',
        marker_line_width=1.5,
        opacity=0.9
    )
    return fig



#########
### TAB 3
#########

def getMostSimilarPortfolios(portfolio_category, portfolio):
    return portfolio_category[str(portfolio)].sort_values(ascending=False).head(10)

def queryRankLift_source(Mama_id):
    source = df_s_reco[(df_s_reco['movie_A']==Mama_id) | (df_s_reco['movie_B']==Mama_id)].sort_values(by='lift',ascending=False).head(50)
    return source

def queryRankLift_country(Mama_id):
    source = df_c_reco[(df_c_reco['movie_A']==Mama_id) | (df_c_reco['movie_B']==Mama_id)].sort_values(by='lift',ascending=False).head(50)
    return source

def queryRankLift_sourcecountry(Mama_id):
    source = df_sc_reco[(df_sc_reco['movie_A']==Mama_id) | (df_sc_reco['movie_B']==Mama_id)].sort_values(by='lift',ascending=False).head(50)
    return source

def getPortfoliosIncludingSingles_source(Mama_id):
    
    querytest = queryRankLift_source(Mama_id=Mama_id)
    
    singles = list(set(list(querytest['movie_A'])+list(querytest['movie_B'])))
    sing_port = dict()
    singles.remove(Mama_id)
    
    for i in range(len(singles)):
        sing_port[singles[i]] = []
        for j in range(len(s_portfolio)):
            if singles[i] in s_portfolio[j]:
                res = s_portfolio.index[j]
                sing_port[singles[i]].append(res)
                
    sing_port = pd.Series(sing_port)
    return sing_port

def getPortfoliosIncludingSingles_country(Mama_id):
    
    querytest = queryRankLift_country(Mama_id=Mama_id)
    
    singles = list(set(list(querytest['movie_A'])+list(querytest['movie_B'])))
    sing_port = dict()
    singles.remove(Mama_id)
    
    for i in range(len(singles)):
        sing_port[singles[i]] = []
        for j in range(len(c_portfolio)):
            if singles[i] in c_portfolio[j]:
                res = c_portfolio.index[j]
                sing_port[singles[i]].append(res)
                
    sing_port = pd.Series(sing_port)
    return sing_port

def getPortfoliosIncludingSingles_sourcecountry(Mama_id):
    
    querytest = queryRankLift_sourcecountry(Mama_id=Mama_id)
    
    singles = list(set(list(querytest['movie_A'])+list(querytest['movie_B'])))
    sing_port = dict()
    singles.remove(Mama_id)
    
    for i in range(len(singles)):
        sing_port[singles[i]] = []
        for j in range(len(sc_portfolio)):
            if singles[i] in sc_portfolio[j]:
                res = sc_portfolio.index[j]
                sing_port[singles[i]].append(res)
                
    sing_port = pd.Series(sing_port)
    return sing_port

def getPortfoliosIncludingCouples_source(Mama_id):
    
    querytest = queryRankLift_source(Mama_id=Mama_id)
    
    couples = []
    coup_port = dict()


    for row in querytest.iterrows():
        res = [int(row[1][0]),int(row[1][1])]
        couples.append(res)

    for i in range(0,len(couples)):
        coup_port[str(couples[i])] = []
        for j in range(len(s_portfolio)):
            if set(couples[i]).issubset(set(s_portfolio[j])):
                res2 = s_portfolio.index[j]
                coup_port[str(couples[i])].append(res2)
    coup_port = pd.Series(coup_port)
    
    # Reindexing couple index to single index withtout queried Mama_id to easily compare single and double dictionaries
    new_index = []

    for index in coup_port.index:
        index = index.replace('[',"").replace(']',"")
        index = [int(s) for s in index.split(',')]
        index.remove(Mama_id)
        new_index.extend(index)

        
    coup_port.index = new_index
    
    return coup_port

def getPortfoliosIncludingCouples_country(Mama_id):
    
    querytest = queryRankLift_country(Mama_id=Mama_id)
    
    couples = []
    coup_port = dict()


    for row in querytest.iterrows():
        res = [int(row[1][0]),int(row[1][1])]
        couples.append(res)

    for i in range(0,len(couples)):
        coup_port[str(couples[i])] = []
        for j in range(len(c_portfolio)):
            if set(couples[i]).issubset(set(c_portfolio[j])):
                res2 = c_portfolio.index[j]
                coup_port[str(couples[i])].append(res2)

    # Series because dictionary keys are hardly alterable and long live Pandas
    coup_port = pd.Series(coup_port)
    
    # Reindexing couple index to single index withtout queried Mama_id to easily compare single and double dictionaries
    
    new_index = []

    for index in coup_port.index:
        index = index.replace('[',"").replace(']',"")
        index = [int(s) for s in index.split(',')]
        index.remove(Mama_id)
        new_index.extend(index)
        
    coup_port.index = new_index
    
    return coup_port

def getPortfoliosIncludingCouples_sourcecountry(Mama_id):
    
    querytest = queryRankLift_sourcecountry(Mama_id=Mama_id)
    
    couples = []
    coup_port = dict()


    for row in querytest.iterrows():
        res = [int(row[1][0]),int(row[1][1])]
        couples.append(res)

    for i in range(0,len(couples)):
        coup_port[str(couples[i])] = []
        for j in range(len(sc_portfolio)):
            if set(couples[i]).issubset(set(sc_portfolio[j])):
                res2 = sc_portfolio.index[j]
                coup_port[str(couples[i])].append(res2)
    # Series because dictionary keys are hardly alterable and long live Pandas
    coup_port = pd.Series(coup_port)
    
    # Reindexing couple index to single index withtout queried Mama_id to easily compare single and double dictionaries
    
    new_index = []

    for index in coup_port.index:
        index = index.replace('[',"").replace(']',"")
        index = [int(s) for s in index.split(',')]
        index.remove(Mama_id)
        new_index.extend(index)
    coup_port.index = new_index
    return coup_port

def getPotentialPlatform_source(Mama_id):
    
    singles = dict(getPortfoliosIncludingSingles_source(Mama_id).sort_index(ascending=True))
    doubles = dict(getPortfoliosIncludingCouples_source(Mama_id).sort_index(ascending=True))
    
    intersect = dict()
    for key in singles.keys():
        singles_key = set(singles[key])
        doubles_key = set(doubles[key])
        res = singles_key - doubles_key
        intersect[key] = res  
    return intersect

def getPotentialPlatform_country(Mama_id):
    
    singles = dict(getPortfoliosIncludingSingles_country(Mama_id).sort_index(ascending=True))
    doubles = dict(getPortfoliosIncludingCouples_country(Mama_id).sort_index(ascending=True))
    
    intersect = dict()
    for key in singles.keys():
        singles_key = set(singles[key])
        doubles_key = set(doubles[key])
        res = singles_key - doubles_key
        intersect[key] = res
    return intersect

def getPotentialPlatform_sourcecountry(Mama_id):
    
    singles = dict(getPortfoliosIncludingSingles_sourcecountry(Mama_id).sort_index(ascending=True))
    doubles = dict(getPortfoliosIncludingCouples_sourcecountry(Mama_id).sort_index(ascending=True))
    
    intersect = dict()
    for key in singles.keys():
        singles_key = set(singles[key])
        doubles_key = set(doubles[key])
        res = singles_key - doubles_key
        intersect[key] = res
    return intersect


def getUniqueMovies(association_platform):
    if association_platform == 'sources':
        df_X_reco = df_s_reco
    elif association_platform == 'countries':
        df_X_reco = df_c_reco
    else:
        df_X_reco = df_sc_reco
    x = list(df_X_reco.movie_A.unique()) + list(df_X_reco.movie_B.unique())
    x = pd.DataFrame(x).drop_duplicates().rename(columns={0:'Mama_id'}).set_index('Mama_id')
    x = x.merge(mama2title,left_index=True, right_index=True, how='left')
    return x


def rankRecosByTransactions_source(title):
    Mama_id = title2mama.loc[title].iloc[0]

    recos = getPotentialPlatform_source(Mama_id)
    recos_list = []
    for key in recos.keys():
        recos_list.append(key)
    recos_by_trans_s  = df1_s[df1_s.index.isin(recos_list)]
    recos_by_trans_s = recos_by_trans_s.groupby('Mama_id')['Transactions'].sum().sort_values(ascending=False)
    
    recos = pd.DataFrame.from_dict(recos, orient='index')
    res = recos.join(recos_by_trans_s, how='left')
    res = res.join(mama2title, how='left')
    n_recos = len(res.columns) - 2
    recos = res.columns[:n_recos]
    res['Opportunities'] = res[recos].values.tolist()
    df1 = res[['Title', 'Transactions']]
    df2 = res[['Title', 'Opportunities']]
    df2 = pd.DataFrame(df2.Opportunities.tolist(), index=df2.Title)
    df2 = df2.stack()
    df2 = df2.reset_index(level=1, drop=True)
    df2 = pd.DataFrame(df2).reset_index()
    df2 = df2.merge(df1, on='Title')
    df2.columns = ['Title', 'Opportunities', 'Transactions']
    df2 = df2.sort_values('Transactions', ascending=False)
    fig = go.Figure(data=[go.Table(
    header=dict(values=['Title', 'Opportunities', 'Transactions'],
                fill_color='rgb(100, 100, 100)',
                font = dict(color = 'white', size = 15),
                align='center',
               line_color='rgb(255, 86, 65)'),
    cells=dict(values=[df2.Title, df2.Opportunities, df2.Transactions],
               align=['left', 'center', 'center'],
               fill_color= 'rgb(60, 60, 60)',
               font = dict(color = 'white', size = 11),
               line_color='rgb(255, 86, 65)'))])

    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
    fig.update_layout(
        width=600,
        height=600)
    return fig

def rankRecosByTransactions_country(title):
    Mama_id = title2mama.loc[title].iloc[0]

    recos = getPotentialPlatform_country(Mama_id)
    recos_list = []
    for key in recos.keys():
        recos_list.append(key)
    recos_by_trans_c  = df1_c[df1_c.index.isin(recos_list)]
    recos_by_trans_c = recos_by_trans_c.groupby('Mama_id')['Transactions'].sum().sort_values(ascending=False)
    
    recos = pd.DataFrame.from_dict(recos, orient='index')
    res = recos.join(recos_by_trans_c, how='left')
    res = res.join(mama2title, how='left')

    n_recos = len(res.columns) - 2
    recos = res.columns[:n_recos]
    res['Opportunities'] = res[recos].values.tolist()
    df1 = res[['Title', 'Transactions']]
    df2 = res[['Title', 'Opportunities']]
    df2 = pd.DataFrame(df2.Opportunities.tolist(), index=df2.Title)
    df2 = df2.stack()
    df2 = df2.reset_index(level=1, drop=True)
    df2 = pd.DataFrame(df2).reset_index()
    df2 = df2.merge(df1, on='Title')
    df2.columns = ['Title', 'Opportunities', 'Transactions']
    df2 = df2.sort_values('Transactions', ascending=False)
    fig = go.Figure(data=[go.Table(
    header=dict(values=['Title', 'Opportunities', 'Transactions'],
                fill_color='rgb(100, 100, 100)',
                font = dict(color = 'white', size = 15),
                align='center',
               line_color='rgb(255, 86, 65)'),
    cells=dict(values=[df2.Title, df2.Opportunities, df2.Transactions],
               align=['left', 'center', 'center'],
               fill_color= 'rgb(60, 60, 60)',
               font = dict(color = 'white', size = 11),
               line_color='rgb(255, 86, 65)'))])

    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
    fig.update_layout(
        width=600,
        height=600)
    return fig

def rankRecosByTransactions_sourcecountry(title):
    Mama_id = title2mama.loc[title].iloc[0]

    recos = getPotentialPlatform_sourcecountry(Mama_id)
    recos_list = []
    for key in recos.keys():
        recos_list.append(key)
    recos_by_trans_sc  = df1_sc[df1_sc.index.isin(recos_list)]
    recos_by_trans_sc = recos_by_trans_sc.groupby('Mama_id')['Transactions'].sum().sort_values(ascending=False)
    
    recos = pd.DataFrame.from_dict(recos, orient='index')
    res = recos.join(recos_by_trans_sc, how='left')
    res = res.join(mama2title, how='left')
    n_recos = len(res.columns) - 2
    recos = res.columns[:n_recos]
    res['Opportunities'] = res[recos].values.tolist()
    df1 = res[['Title', 'Transactions']]
    df2 = res[['Title', 'Opportunities']]
    df2 = pd.DataFrame(df2.Opportunities.tolist(), index=df2.Title)
    df2 = df2.stack()
    df2 = df2.reset_index(level=1, drop=True)
    df2 = pd.DataFrame(df2).reset_index()
    df2 = df2.merge(df1, on='Title')
    df2.columns = ['Title', 'Opportunities', 'Transactions']
    df2 = df2.sort_values('Transactions', ascending=False)
    fig = go.Figure(data=[go.Table(
    header=dict(values=['Title', 'Opportunities', 'Transactions'],
                fill_color='rgb(100, 100, 100)',
                font = dict(color = 'white', size = 15),
                align='center',
               line_color='rgb(255, 86, 65)'),
    cells=dict(values=[df2.Title, df2.Opportunities, df2.Transactions],
               align=['left', 'center', 'center'],
               fill_color= 'rgb(60, 60, 60)',
               font = dict(color = 'white', size = 11),
               line_color='rgb(255, 86, 65)'))])

    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
    fig.update_layout(
        width=600,
        height=600)
    return fig

# Display embeddings
def display_emb(df=df_tab3_2, tags=tags_3,query=''):

    # Embedding dataframe
    dft = df[['Mama_id', 'Title']].drop_duplicates()
    dft.set_index('Mama_id', inplace=True)
    dft = tags.merge(dft, on='Mama_id', how='left')
    data = pd.DataFrame(movie2vec, index=dft.index)
    data['Title'] = dft['Title']
    trans = df.groupby('Title')['Royalties'].sum().reset_index()
    data = data.merge(trans, on='Title', how='left')
    data.columns = [0, 1, 'Title', 'Trans']
    data = data.sort_values('Trans',ascending=False)
    data = data[data['Title'].notna()]
    data['Size'] = 30
        
    if query != '':
        # Plot the 10 points that are most closed to the queries movie
        x = data[data['Title'] == query][0].values[0]
        y = data[data['Title'] == query][1].values[0]
        data['d'] = (data[0]-x) ** 2 + (data[1]-y) ** 2
        data = data.sort_values('d')
        data = data.iloc[:11]
        data['Size'].iloc[0] = 20

    # Plot results
    fig = go.Figure(data=go.Scatter(x=data[0],
                                    y=data[1],
                                    mode='markers',
                                    marker_color=data['Trans'],
                                    marker_size=data['Size'],
                                    text=data['Title']))

    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)'
    })
    
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_traces(
        marker_color= 'rgb(255, 86, 65)')
    return fig



