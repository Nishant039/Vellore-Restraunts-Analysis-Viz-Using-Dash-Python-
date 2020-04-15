from dash.dependencies import Input,Output
import dash
import dash_core_components as dcc
import dash_html_components as html
from plotly.offline import plot
import plotly.graph_objs as go
import numpy as np
import plotly
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use("ggplot")
import re
import random
import plotly.express as px

df=pd.read_csv("Vellore Restaurants Survey-2.csv")
dic={i:j for (i,j) in zip(df.Restaurant.value_counts().index,df.Restaurant.value_counts().values)}
df["Popularity"]=df["Restaurant"].map(dic)

loc=[re.findall(r"[(](.*)[)]",i)[0] for i in df["Restaurant"]]

df["location"]=loc


top1Div_data=[go.Indicator(
    mode = "gauge+number",
    value = df.shape[0],
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = {'text': "Number of Surveys Conducted"},
gauge={
    'bgcolor':'black',
    'bordercolor':'gray',
    'bar':{'color':'#00FF00'}
})]

top1Div_layout=go.Layout(paper_bgcolor='#222222',plot_bgcolor='#222222',font={'color':'white'},
                        height=320)


labels=['Hosteller','Day Scholar']
values=pd.Series([1 if x=='Yes' else 0 for x in df['Are You a Hosteller?']])
values=[values.value_counts()[1],values.value_counts()[0]]


top2Div_data=[go.Pie(labels=labels, values=values, hole=.7,
                hoverinfo='label+percent', textinfo='value', textfont_size=20,
                  title_text="People Surveyed",
                  marker=dict(line=dict(color='#000000', width=2)))]
top2Div_layout=go.Layout(paper_bgcolor='#222222',plot_bgcolor='#222222',font={'color':'white'},
                        height=320)



top1Div=html.Div([

    dcc.Graph(id='top1div',figure={
        'data':top1Div_data,
        'layout':top1Div_layout
    })
],style={'background-color':'#222222','min-width':'23%','height':'50px','display':'inline-block','margin':'10px'})



top2Div=html.Div([
    dcc.Graph(id='top2div',
             figure={
                 'data':top2Div_data,
                 'layout':top2Div_layout
             })
],style={'background-color':'#222222','min-width':'23%','height':'50px','display':'inline-block','margin':'10px'})



options_top3div=[]
for i in df.drop(["Please write your favorite dish from this restaurant ","Best cuisines of the restaurant"],axis=1).columns:
    if(df[i].dtype=='object'):
        options_top3div.append({'label':i.strip(),'value':i})
top3Div=html.Div([
    dcc.Dropdown(id='uni_cat',
                options=options_top3div,
                value='How crowded is the restaurant usually?',
                style={'color':'black','background-color':'#ECD9B8'}),
    dcc.Graph(id='graph_uni_cat')


],style={'background-color':'#222222','min-width':'23%','height':'50px','display':'inline-block','margin':'10px'})



x=df.Restaurant.value_counts().index
y=df.Restaurant.value_counts().values

midDiv_data=[go.Bar(x=x, y=y ,text='Surveys',marker_color='rgb(26, 118, 255)')]

midDiv_layout=go.Layout(yaxis=dict(showgrid=False,automargin=True),
                        xaxis=dict(automargin=True),
                        title_text='Popularity of Restaurants in Survey',xaxis_tickangle=-45,
                        paper_bgcolor='#222222',plot_bgcolor='#222222',font={'color':'white'},
                       height=500)

midDiv=html.Div([
    dcc.Graph(id='midgraph',
             figure={
                 'data':midDiv_data,
                 'layout':midDiv_layout
             })
])


options_downmidDiv=[]
for i in df["Restaurant"]:
    options_downmidDiv.append({'label':i,'value':i})


downmidDiv1=html.Div([
    dcc.Dropdown(id='res_rating',
             options=options_downmidDiv,
             value='FLO CAFE (VIT MAIN GATE)',style={'background-color':'#ECD9B8',
                'color':'black','vertical-align':'top'}),
    dcc.Graph(id='rating_graph')




],style={'background-color':'#222222','min-width':'40%','display':'inline-block'})



words=[]
for i in df["Best cuisines of the restaurant"]:
    temp=i.split(";")
    for j in temp:
        words.append(j)
word_counts=pd.Series(words).value_counts()
weights=list(word_counts.values)
words_unique=list(word_counts.index)

weights,words_unique
words = words_unique
colors = [plotly.colors.DEFAULT_PLOTLY_COLORS[random.randrange(1, 10)] for i in range(5)]
weights = weights



data = go.Scatter(x=[random.random() for i in range(30)],
                 y=[random.random() for i in range(30)],
                 mode='text',
                 text=words,
                 marker={'opacity': 0.3},
                 textfont={'size': weights,
                           'color': colors})
layout = go.Layout({'xaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False},
                    'yaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False},
                   'title':{'text':"Word CLoud For Cuisines  (Please Regenerate if not clear)",'font':{'color':'white'}}
                   ,'paper_bgcolor':'#222222','plot_bgcolor':'#222222',
                   'height':420})
fig = go.Figure(data=[data], layout=layout)


downmidDiv2=html.Div([
    dcc.Graph(id='word_cloud',
             figure=fig)
],style={'background-color':'#222222','min-width':'50%','display':'inline-block','margin':'30px','vertical-align':'top'})

cat_optionsdownDiv=[]
for i in df.drop(["Please write your favorite dish from this restaurant ","Best cuisines of the restaurant"],axis=1).columns:
    if(df[i].dtype=='object'):
        cat_optionsdownDiv.append({'label':i.strip(),'value':i})

num_optionsdownDiv=[]
for i in df.drop(["Please write your favorite dish from this restaurant ","Best cuisines of the restaurant"],axis=1).columns:
    if(df[i].dtype!='object'):
        num_optionsdownDiv.append({'label':i.strip(),'value':i})

downDiv1=html.Div([
    dcc.Dropdown(id='bi_cat_num_cat',
                options=cat_optionsdownDiv,
                value='How crowded is the restaurant usually?',style={'color':'black','background-color':'#ECD9B8'}),
    dcc.Dropdown(id='bi_cat_num_num',
                options=num_optionsdownDiv,
                value='Rating',style={'color':'black','background-color':'#ECD9B8'}),

    dcc.Graph(id='bi_cat_num_graph')
],style={'background-color':'#121212','min-width':'45%','display':'inline-block','margin':'30px','vertical-align':'top'})

cuisine_options=[]
for i in words_unique:
    cuisine_options.append({'label':i,'value':i})
downDiv2=html.Div([

    dcc.Dropdown(id='cuisine-options',
    options=cuisine_options,
    value="North Indian",style={'background-color':'#ECD9B8','min-width':'40%','display':'inline-block'}),


    dcc.Graph(id="top_cuisine")



],style={'background-color':'#121212','min-width':'45%','display':'inline-block','margin':'30px','vertical-align':'top'})


heatDiv=html.Div([

    html.Div([

    dcc.Dropdown(id='heatcat1',
                options=cat_optionsdownDiv,
                value=cat_optionsdownDiv[0]['label'],style={'background-color':'#ECD9B8','min-width':'27%','display':'inline-block',
                                                  'margin':'1px'}),

    dcc.Dropdown(id='heatcat2',
                options=cat_optionsdownDiv,
                value='location',style={'background-color':'#ECD9B8','min-width':'27%','display':'inline-block','margin':'1px'}),


    dcc.Dropdown(id='heatnum',
                options=num_optionsdownDiv,
                value='Rating',style={'background-color':'#ECD9B8','min-width':'27%','display':'inline-block','margin':'1px'}),

    dcc.Graph(id='heatmap')



    ],style={'margin':'0px auto','text-align':'center'})


])

bottomDiv1=html.Div([

    dcc.Dropdown(id='scatternum1',
                options=num_optionsdownDiv,
                value='Popularity',
                style={'background-color':'#ECD9B8','min-width':'27%','display':'inline-block',
                                                  'margin':'1px'}),

    dcc.Dropdown(id='scatternum2',
                options=num_optionsdownDiv,
                value='Rating',
                style={'background-color':'#ECD9B8','min-width':'27%','display':'inline-block',
                                                  'margin':'1px'}),

    dcc.Dropdown(id='scattercat',
                options=cat_optionsdownDiv,
                value='location',
                style={'background-color':'#ECD9B8','min-width':'27%','display':'inline-block',
                                                  'margin':'1px'}),

    dcc.Graph(id='scatter')



],style={'background-color':'#121212','min-width':'45%','display':'inline-block','margin':'30px','vertical-align':'top'})
bottomDiv2=html.Div([
    dcc.Dropdown(id='regnum1',
               options= num_optionsdownDiv,
                value='Rating',style={'background-color':'#ECD9B8','min-width':'40%','display':'inline-block',
                                                  'margin':'1px'}),
    dcc.Dropdown(id='regnum2',
                options=num_optionsdownDiv,
                value='Popularity',style={'background-color':'#ECD9B8','min-width':'40%','display':'inline-block',
                                                  'margin':'1px'}),
    dcc.Graph(id='reg')
],style={'background-color':'#121212','min-width':'45%','display':'inline-block','margin':'30px','vertical-align':'top'})

lastDiv=html.Div([

    dcc.Dropdown(id='num',
                options=num_optionsdownDiv,
                value='Popularity',
                style={'background-color':'#ECD9B8','min-width':'40%',
                                                  'margin':'1px'}),
    dcc.Graph(id='hist')
],style={'background-color':'#121212','min-width':'80%','display':'inline-block','margin':'30px','vertical-align':'top'})

external_stylesheets = ['https://fonts.googleapis.com/css?family=Girassol']
app=dash.Dash(__name__,external_stylesheets=external_stylesheets)
server=app.server

app.layout=html.Div([

    html.Div(html.H1("Vellore Restaurants Dashboard"),style={'color':'white','font-family':'Girassol',
                                                             'text-align':'left',
                                                            'fontSize':'0.7em'}),
    html.Div([
    top1Div,
    top2Div,
    top3Div
    ],style={'text-align':'center','width':'100%','min-height':'400px'}),


    html.Div([

        midDiv

    ],style={'width':'100%','min-height':'550px','background-color':'#222222'}),


    html.Div([

        downmidDiv2,
        downmidDiv1



    ],style={'min-height':'550px','text-align':'center','marginTop':'80px'}),


    html.Div([
        downDiv1,
        downDiv2

    ],style={'min-height':'550px','background-color':'#222222','width':'100%'}),




    html.Div([

        heatDiv

    ],style={'min-height':'550px','width':'100%','marginTop':'80px'}),



    html.Div([

        bottomDiv1,
        bottomDiv2

    ],style={'min-height':'550px','background-color':'#222222','width':'100%'}),


    html.Div([

        lastDiv

    ],style={'min-height':'400px','background-color':'#222222','max-width':'800px','margin':'80px auto'}),


    html.Div([
        "© Developer - Nishant Pandey",
        html.Hr()
        ,
        dcc.Link("Github",href="https://github.com/Nishant039",style={'text-decoration':'none','color':'black'})
    ],style={'min-height':'30px','background-color':'white'})



],style={'margin':'0px','background-color':'#121212','padding':'0px','height':'100%','marginBottom':'20px'})

def most_frequent(List):
    List=list(List)
    return max(set(List), key = List.count)







### Callbacks

@app.callback(Output('graph_uni_cat','figure'),[Input('uni_cat','value')])
def update_uni_cat(cat):
    df_cat_group=df.groupby("Restaurant").agg({cat:most_frequent})
    values=list(df_cat_group[cat].value_counts().values)
    labels=list(df_cat_group[cat].value_counts().index)


    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.7)])
    fig.update_layout(title={'text':cat},paper_bgcolor='#222222',plot_bgcolor='#222222',font={'color':'white'},height=285)
    return fig



@app.callback(Output('rating_graph','figure'),[Input('res_rating','value')])
def update_rating(resName):
    df_res=df[df["Restaurant"]==resName]
    value=list(df_res["Rating"].value_counts().values)
    rating=list(df_res["Rating"].value_counts().index)
    expected=[1,2,3,4,5]
    for i in expected:
        if i not in rating:
            rating.append(i)
            value.append(0)

    fig = go.Figure(go.Bar(
            x=value,
            y=rating,marker={
            'line':{'color':'white','width':1},
            'color':'#00FF00',
            'opacity':0.5},
            orientation='h',hovertemplate = '%{y} star ratings : %{x}<extra></extra>'))

    fig.update_layout(
        font=dict(
        size=18,
        color="#FFFFFF"
    ),

    title={'text':"Ratings for {}".format(resName),
          'font':{'size':25}},
    yaxis=dict(
        showgrid=False,
        showline=False,
        showticklabels=True
    ),
    xaxis=dict(
        zeroline=False,
        showline=False,
        showticklabels=True,
        showgrid=False
    ),paper_bgcolor='#222222',
    plot_bgcolor='#222222',
    height=410)

    return fig



@app.callback(Output('bi_cat_num_graph','figure'),[Input('bi_cat_num_num','value'),Input('bi_cat_num_cat','value')])
def make_biv_cat_num(num1,cat):
    traces=[]
    for category in df[cat].unique():
        df_cat=df[df[cat]==category]

        traces.append(go.Box(y=df_cat[num1],name=category))

    fig=go.Figure(data=traces)
    fig.update_layout(yaxis={'title':num1,'showgrid':False,'automargin':True},xaxis={'title':cat,'showgrid':False,'automargin':True},paper_bgcolor='#121212',plot_bgcolor='#121212',font={'color':'white'},
                     hovermode='closest',height=400)
    return fig



@app.callback(Output('top_cuisine','figure'),[Input('cuisine-options','value')])
def cusine_plot(cuisine):
    res_names=[]
    for i in df["Best cuisines of the restaurant"]:
        if(cuisine in i.split(";")):
            res_names.append(True)
        else:
            res_names.append(False)
    rests=df["Restaurant"][res_names].unique()
    df_grp=df.groupby("Restaurant").agg({'Rating':np.mean})
    df_grp=df_grp.loc[rests].sort_values(by='Rating',ascending=False)
    x=df_grp.index
    y=df_grp.values.ravel()
    fig = go.Figure(data=[go.Scatter(x=x, y=y ,mode='lines+markers',text='Cuisine and rating')])
    fig.update_layout(hovermode='closest',title={'text':"Top Restaurants for cusuine "+cuisine+"(with ratings)"},font={'color':'white'}
                   ,paper_bgcolor='#121212',plot_bgcolor='#121212',xaxis={'showgrid':False,'automargin':True},yaxis={'showgrid':False,'automargin':True},height=430,xaxis_tickangle=-45)
    return fig



@app.callback(Output('heatmap','figure'),[Input('heatcat1','value'),Input('heatcat2','value'),Input('heatnum','value')])
def heatmap(heatcat1,heatcat2,heatnum):
    temp=df[[heatcat1,heatcat2,heatnum]].set_index(heatcat1)
    temp=temp.pivot_table(values=heatnum,columns=heatcat2,index=heatcat1)
    z=temp.values
    x=temp.columns
    y=temp.index
    fig = go.Figure(data=go.Heatmap(
                   x=x,y=y,z=z,
                   hoverongaps = False))
    fig.update_layout(yaxis={'title':heatcat2,'showgrid':False},xaxis={'title':heatcat1,'showgrid':False},paper_bgcolor='#121212',plot_bgcolor='#222222',font={'color':'white'},
                     hovermode='closest',title={'text':heatnum+" (Includes NaN values)"})

    return fig





@app.callback(Output('scatter','figure'),[Input('scatternum1','value'),Input('scatternum2','value'),Input('scattercat','value')])
def scatter(num1,num2,cat):
    x=df[num1]
    y=df[num2]
    c=df[cat]
    traces=[]
    for category in df[cat].unique():
        df_cat=df[df[cat]==category]
        x=df_cat[num1]
        y=df_cat[num2]

        traces.append(go.Scatter(x=x,y=y,mode='markers',name=category,
    marker={'size':15,'opacity':0.7}))



#     print(x)
    fig = go.Figure(data=traces)

    fig.update_layout(
    title=cat,
    xaxis={'title':num1,'showgrid':False},
    yaxis={'title':num2,'showgrid':False},
    paper_bgcolor='#121212',plot_bgcolor='#222222',font={'color':'white'})

    return fig






@app.callback(Output('reg','figure'),[Input('regnum1','value'),Input('regnum2','value')])
def reg(num2,num1):
    df_group=df.groupby("Restaurant").agg({num1:np.mean,
                                          num2:np.mean})

    df_group.reset_index(inplace=True)

    fig = px.scatter(df_group, x=num1, y=num2, trendline="ols")
    fig.update_traces(marker=dict(size=15,opacity=0.7))
    fig.update_layout(xaxis={'showgrid':False},yaxis={'showgrid':False},paper_bgcolor='#121212',plot_bgcolor='#222222',font={'color':'white'},
                     hovermode='closest')

    return fig


@app.callback(Output('hist','figure'),[Input('num','value')])
def hist(num):
    y=df[num]
    fig = go.Figure(data=[go.Histogram(x=y)])
    fig.update_layout(title={'text':num},xaxis={'showgrid':False},yaxis={'showgrid':False},paper_bgcolor='#121212',plot_bgcolor='#222222',font={'color':'white'},
                     hovermode='closest')

    return fig


app.css.config.serve_locally = True
app.scripts.config.serve_locally = True
if(__name__=='__main__'):
    app.run_server()
