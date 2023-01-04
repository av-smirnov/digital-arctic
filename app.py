import re
from urllib.parse import unquote

import dash
from dash import html
from dash import dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input, State
from dash.exceptions import PreventUpdate
from dash import dash_table
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import json

globalbgcolor = '#f7fbff'
app = dash.Dash(__name__, 
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0, maximum-scale=4, minimum-scale=0.5,'}],
                external_stylesheets=[dbc.themes.BOOTSTRAP])  # BOOTSTRAP COSMO PULSE ZEPHYR MATERIA LITERA
server = app.server 


profiles = pd.read_csv('data\profiles.csv', delimiter = ';', low_memory=False)
molist = pd.read_csv('data\molist.csv', delimiter = ';', low_memory=False)
areas = molist['Территория'].drop_duplicates().tolist()
stats = pd.read_csv('data/stats.csv', delimiter = ';', low_memory=False)
agesex = pd.read_csv('data/age_2021.csv', delimiter = ';', low_memory=False)
transport = pd.read_csv("data/transport.csv", sep=',')
transport_cit = pd.read_csv("data/transport_cities.csv", sep=',')
urban = pd.read_csv('data/urban.csv', delimiter = ';')
education = pd.read_csv('data/education.csv', delimiter = ';')
tidy = pd.read_csv('data/tidy.csv', delimiter = ';')
indicators_info = pd.read_csv('data/indicators.csv', delimiter = ';')
coords = pd.read_csv('data/coords.csv', delimiter = ';')
dff = pd.read_csv("data/mumcolor2.csv", sep=';')
with open('data/rusmo10_10.geojson', encoding='utf-8') as json_file:
    rusmo = json.load(json_file)
cividis0 = px.colors.sequential.Cividis[0]



def make_empty_fig():
    fig = go.Figure()
    fig.layout.paper_bgcolor = globalbgcolor
    fig.layout.plot_bgcolor = globalbgcolor
    return fig


def multiline_indicator(indicator):
    final = []
    split = indicator.split()
    for i in range(0, len(split), 3):
        final.append(' '.join(split[i:i+3]))
    return '<br>'.join(final)


main_layout = html.Div([
    html.Div([
    dbc.NavbarSimple([
        dbc.DropdownMenu([
                dbc.DropdownMenuItem(area, href=area) for area in areas
            ], label='Выберите территорию'),
        ], brand='Главная страница',brand_href='/'),
    dcc.Location(id='location'),
    html.Div(id='main_content'),
    html.Br(),
    dbc.Row([
        dbc.Col(lg=1),
        dbc.Col([
            dbc.Tabs([            
                dbc.Tab([
                    html.Ul([
                        html.Br(),
                        html.Li([
                            'Дашборд разработан в рамках гранта ', html.B('Российского научного фонда'), ' № 21-78-00081. Сайт проекта: ',
                            html.A('https://arcdem.ru',
                                    href='https://arcdem.ru')
                                ]),
                        html.Li([
                            'Руководитель проекта и автор сайта – к.э.н., с.н.с. лаборатории демографии и социального управления '
                            'ИСЭ и ЭПС ФИЦ Коми НЦ УрО РАН ', html.B('Андрей Владимирович Смирнов'), ' (',
                            html.A('av.smirnov.ru@gmail.com',
                                   href='mailto:av.smirnov.ru@gmail.com'), ')'
                        ]),
                        html.Li([
                            'Репозиторий на GitHub: ',
                            html.A('https://github.com/av-smirnov/digital-arctic',
                                   href='https://github.com/av-smirnov/digital-arctic')
                        ]),
                        html.Li('Последнее обновление: 30.12.2022')
                    ])
                ], label='О проекте'),
                    dbc.Tab([
                        html.Ul([
                            html.Br(),
                            html.Li([
                            'Научные статьи о населении Арктики: ',
                            html.A('http://vvfauzer.ru/index/arctic/0-18',
                                   href='http://vvfauzer.ru/index/arctic/0-18')
                        ])
                        ])
                    ], label='Полезные ссылки')
                ]),
            ])
    ])
], style={ 'backgroundColor': globalbgcolor
     })
])

area_dashboard = html.Div([
    dbc.Row([
        dbc.Col(lg=1),
        dbc.Col([
    html.Br(),
        html.H1(id='country_heading'),
        dbc.Row([
            dbc.Col(dcc.Graph(id='country_page_graph'))
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Label('Выберите показатель:'),
                dcc.Dropdown(id='country_page_indicator_dropdown',
                             placeholder='Выберите показатель',
                             value='Среднегодовая численность населения, человек',
                             options=[{'label': indicator, 'value': indicator}
                                     for indicator in tidy.columns[9:65]]),
            ], lg=6, md=11),
            dbc.Col([
                dbc.Label('Выберите территории:'),
                dcc.Dropdown(id='country_page_contry_dropdown',
                             placeholder='Выберите одну или несколько территорий для сравнения',
                             multi=True,
                             options=[{'label': c, 'value': c}
                                       for c in areas]),
            ], lg=6, md=11)
        ]),
        html.Br(),
        html.H4('Расселение городского населения'),
        html.Div(id='urban_table'),
        html.Br(), html.Br(),
        dcc.Graph(id='age_pyramid'),
        html.Br(), html.Br(),
        dcc.Graph(id='edu_graph'),
        html.Br(), html.H4('Основные демографические показатели'),
        html.Div(id='country_table')
        ], lg=10)
    ]),
])

indicators_dashboard = html.Div([
    dbc.Col([
        html.Br(),
        html.H1('Цифровой двойник населения Арктики'),

    ], style={'textAlign': 'center'}),
    html.Br(),
    dbc.Row([
        dbc.Col(lg=2),
        dbc.Col([
            dbc.Tabs([
                dbc.Tab([
                    html.Br(),
                    dbc.Label('Выберите показатель:'),
                    dcc.Dropdown(id='indicator_dropdown',
                                 value='Естественный прирост (убыль), человек',
                                 options=[{'label': indicator,
                                 'value': indicator} 
                                 for indicator in tidy.columns[9:65]]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label('Выберите год:'),
                            dcc.Slider(id='indicator_map_slider',
                                       min=2010,
                                       max=2021,
                                       step=1,
                                       included=False,
                                       value=2021,
                                       marks={year: {'label': str(year),
                                                     'style': {'color': cividis0, 'fontSize': 14}}
                                              for year in [x for x in range(2010, 2022)]}),
                        ], lg=7),
                        dbc.Col([
                            dbc.Label('Выберите цветовую шкалу:'),
                            dcc.Dropdown(id='indicator_map_color_dropdown',
                                         value='Спектральная',
                                         options=[{'label': indicator,
                                         'value': indicator}
                                         for indicator in ['Спектральная', 'От красного к синему',
                                                           'От красного к зеленому','Viridis','Cividis']]),
                        ], lg=3),
                        dbc.Col([
                            html.Br(), html.Br(),
                            dcc.Checklist(id='indicator_map_check',
                                              options=[
                                                  {'label': 'Обратная шкала', 'value': 'invert'},
                                              ],
                                              value=[]
                                          ),
                        ], lg=2),
                    ]),
                    dcc.Graph(id='indicator_map_chart'),
                    dcc.Markdown(id='indicator_map_details_md',
                                style={'backgroundColor': globalbgcolor}
                         ),
                    dcc.Graph(id='indicator_year_barchart',
                             figure=make_empty_fig())
                ], label='Изучение показателей'),
                dbc.Tab([
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label('Показатель по оси X:'),
                            dcc.Dropdown(id='mma_indicator1_dropdown',optionHeight=40,
                                            value='Отношение среднемесячной заработной платы к стоимости фиксированного набора, раз',
                                            options=[{'label': indicator, 'value': indicator}
                                                    for indicator in tidy.columns[9:65]]),
                        ], lg=8),
                        dbc.Col([
                            html.Br(), html.Br(),
                            dcc.Checklist(id='mma_check1',
                                options=[
                                    {'label': 'Логарифмическая шкала', 'value': 'log'},
                                ],
                                value=[]
                            )
                        ], lg=4),
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label('Показатель по оси Y:'),
                            dcc.Dropdown(id='mma_indicator2_dropdown', optionHeight=40,
                                         value='Коэффициент миграционного прироста (общий)',
                                         options=[{'label': indicator, 'value': indicator}
                                                  for indicator in tidy.columns[9:65]]),
                        ], lg=8),
                        dbc.Col([
                            html.Br(), html.Br(),
                            dcc.Checklist(id='mma_check2',
                                options=[
                                    {'label': 'Логарифмическая шкала', 'value': 'log'},
                                ],
                                value=[]
                            )
                        ], lg=4),
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label('Размер метки:'),
                            dcc.Dropdown(id='mma_indicator3_dropdown',optionHeight=40,
                                            value='Среднегодовая численность населения, человек',
                                            options=[{'label': indicator, 'value': indicator}
                                                    for indicator in ['Среднегодовая численность населения, человек']]),
                        ], lg=6),
                        dbc.Col([
                            dbc.Label('Цвет метки:'),
                            dcc.Dropdown(id='mma_indicator4_dropdown', optionHeight=40,
                                         value='Субъект РФ',
                                         options=[{'label': indicator, 'value': indicator}
                                                  for indicator in ['Субъект РФ', 'Тип', 'Уровень', 'Часть Арктики']]),
                        ], lg=6),
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                                dbc.Label('Выберите год:'),
                                dcc.Slider(id='mma_slider',
                                           min=2010,
                                           max=2021,
                                           step=1,
                                           included=False,
                                           value=2021,
                                           marks={year: {'label': str(year),
                                                         'style': {'color': cividis0, 'fontSize': 14}}
                                                  for year in [x for x in range(2010, 2022)]}),
                                ], lg=12),
                    ]),
                    html.Br(),
                    dcc.Graph(id='mma_graph',figure=make_empty_fig()),
                ], label='Многомерный анализ'),
                dbc.Tab([
                    html.Br(),
                    dbc.Row([
                        dbc.Col(lg=1),
                        dbc.Col([
                            dbc.Label('Выберите год:'),
                            dcc.Slider(id='year_cluster_slider',
                                    min=2010, max=2022, step=1, included=False,
                                    value=2021,
                                    marks={year: str(year)
                                            for year in range(2010, 2023, 2)})
                        ], lg=6, md=12),
                        dbc.Col([
                            dbc.Label('Выберите число кластеров:'),
                            dcc.Slider(id='ncluster_cluster_slider',
                                    min=2, max=10, step=1, included=False,
                                    value=4,
                                    marks={n: str(n) for n in range(2, 11)}),
                        ], lg=4, md=12)
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col(lg=1),
                        dbc.Col([
                            dbc.Label('Выберите индикаторы:'),
                            dcc.Dropdown(id='cluster_indicator_dropdown',optionHeight=40,
                                        multi=True,
                                        value=['Отгружено товаров собственного производства, выполнено работ и услуг, млн рублей'],
                                        options=[{'label': indicator, 'value': indicator}
                                                for indicator in tidy.columns[9:65]]),
                        ], lg=6),
                        dbc.Col([            
                            dbc.Label(''),html.Br(),
                            dbc.Button("Отправить", id='clustering_submit_button'),
                        ]),
                    ]),
                    dcc.Loading([
                        dcc.Graph(id='clustered_map_chart')
                    ])
                ], label='Кластеризация'),
                dbc.Tab([
                    html.Br()
                ], label='Расселение'),
                dbc.Tab([
                    html.Br(),
                    dcc.Dropdown(id='migration_dropdown',
                                 value='Самолеты и поезда',
                                 options=[{'label': param,
                                 'value': param}
                                 for param in ['Самолеты и поезда','Самолеты','Поезда']]),
                    dcc.Loading([
                        dcc.Graph(id='migration_flows')
                    ]),
                    html.Li([
                        html.A(('Полный набор данных на сайте ', html.B('Туту.ру'), ' и его описание'),
                                href='https://story.tutu.ru/dataset-tutu-ru-i-dannye-modeli-open-data-science/')
                            ]),
                    html.Li([
                        'Пример изучения перемещений населения методами сетевого анализа представлен в статье "',
                        html.B(html.A('Цифровые следы населения как источник данных о миграционных потоках в российской Арктике',
                                href='https://www.avsci.ru/p/1_25.pdf')), '"'
                    ])
                ], label='Перемещения'),

                dbc.Tab([
                    html.Br()
                ], label='Наука и образование'),

                dbc.Tab([
                    html.Br()
                ], label='Пандемия'),
            ]),
        ], lg=8)
    ]),
    html.Br(),
] ,  style={'backgroundColor': globalbgcolor}
)

app.validation_layout = html.Div([
    main_layout,
    indicators_dashboard,
    area_dashboard,
])

app.layout = main_layout

@app.callback(Output('main_content', 'children'),
              Input('location', 'pathname'))
def display_content(pathname):
    if unquote(pathname[1:]) in areas:
        return area_dashboard
    else:
        return indicators_dashboard

@app.callback(Output('indicator_map_chart', 'figure'),
              Output('indicator_map_details_md', 'children'),
              Output('indicator_year_barchart', 'figure'),
              Input('indicator_dropdown', 'value'),
              Input('indicator_map_slider', 'value'),
              Input('indicator_map_color_dropdown', 'value'),
              Input('indicator_map_check','value'))
def display_generic_map_chart(indicator, year, mapcolor, invert):
    color_dict = {
        "Спектральная": "spectral",
        "От красного к синему": "RdBu",
        "От красного к зеленому": "RdYlGn",
        "Viridis": "Viridis",
        "Cividis": "Cividis",
    }

    imapcolor = color_dict[mapcolor]
    if 'invert' in invert:
        imapcolor = imapcolor + '_r'

    if indicator is None:
        raise PreventUpdate
   # df = tidy[tidy['Тип'].isin(['городской округ','муниципальный район','муниципальный округ','городской округ (закрытое адм.-тер. образование)']) & tidy['Год'].eq(2020)]

    dat = tidy[tidy['Тип'].isin(['городской округ', 'муниципальный район', 'муниципальный округ',
                                'городской округ (закрытое адм.-тер. образование)'])  & tidy['Год'].eq(year)]

    df = pd.merge(dat, coords, left_on='Территория',right_on='Территория',how='left')
    #fig = px.choropleth(df, geojson=rusmo, featureidkey = "properties.id",
    #                    locations='ОКТМО',
    #                    color=indicator,
    #                    title=indicator,
    #                    hover_name='Территория',
    #                    color_continuous_scale='spectral',
    #                    animation_frame='Год',
    #                    height=750)
    #fig.update_geos(projection_type="conic equal area", projection_rotation_roll=0,
    #                projection_rotation_lat=15,
    #                projection_rotation_lon=105,
    #                center_lat=71,
    #                center_lon=107,
    #                projection_scale=6)
    #fig.layout.geo.showframe = False
    #fig.layout.geo.showcountries = False
    #fig.layout.geo.landcolor = 'white'
    #fig.layout.geo.bgcolor = globalbgcolor
    #fig.layout.paper_bgcolor = globalbgcolor
    #fig.layout.geo.countrycolor = 'gray'
    #fig.layout.geo.coastlinecolor = 'gray'
    #fig.layout.coloraxis.colorbar.title = multiline_indicator(indicator)

    fig = go.Figure()
    fig.add_trace(go.Choropleth(
        geojson=rusmo, featureidkey="properties.id",
        locations=df['ОКТМО'],
        z=df[indicator], hoverinfo='text',
        marker=dict(line_width=0.5, line_color='rgb(140,140,140)'),
        colorscale=imapcolor, showscale=True,
        text= df['Территория'],
        customdata=df[indicator],
        hovertemplate = '%{text}<br>Значение: %{customdata}<extra></extra>',
    ))

    try:
        fig.add_trace(go.Scattergeo(
            lon = df['Долгота'],
            lat = df['Широта'],
            mode='markers',
            marker_size = df['Размер'],
            marker_color = df[indicator],
            text=df['Территория'],
            customdata=df[indicator],
            marker=dict(
                colorscale=imapcolor,
                line=dict(width=1,color='rgb(20, 20, 20)'),
            ),
            hovertemplate='%{text}<br>Значение: %{customdata}<extra></extra>',
        ))
    except Exception:
        aaa = 1



    fig.update_geos(projection_type="conic equal area", projection_rotation_roll=0,
                    projection_rotation_lat=15,
                    projection_rotation_lon=105,
                    center_lat=71,
                    center_lon=107,
                    projection_scale=6,
                    showcountries=True, countrywidth = 0.5, coastlinewidth = 0.5,
                    )
    fig.update_layout(
        height=750,
    )
    fig.layout.geo.bgcolor = globalbgcolor
    fig.layout.paper_bgcolor = globalbgcolor
    fig.layout.geo.landcolor = 'rgb(245, 245, 245)'





    
    series_df = indicators_info[indicators_info['Показатель'].eq(indicator)]
    if series_df.empty:
        markdown = "Нет данных по данному показателю"
    else:
        limitations = series_df['Ограничения и комментарий'].fillna('...').str.replace('\n\n', ' ').values[0]

        markdown = f"""
        ## {series_df['Показатель'].values[0]}  
        
        {series_df['Описание'].values[0]}  
        
        * **Группа показателей:** {series_df['Группа показателей'].fillna('count').values[0]}
        * **Единица измерения:** {series_df['Единица измерения'].fillna('count').values[0]}
        * **Периодичность:** {series_df['Период'].fillna('N/A').values[0]}
        * **Источник:** {series_df['Источник'].values[0]}
        
        ### Ограничения и комментарии:  
        
        {limitations}  

        """

    if not year:
        raise PreventUpdate
    df = tidy[tidy['Год'].eq(year)].sort_values(indicator).dropna(subset=[indicator])
    n_countries = len(df['Территория'])
    fig2 = px.bar(df,
                 x=indicator,
                 color = 'Уровень',
                 y='Территория',
                 orientation='h',
                 height=200 + (n_countries*20),
                 title=indicator + ', ' + str(year))
    fig2.layout.paper_bgcolor = globalbgcolor



    return fig, markdown, fig2


@app.callback(Output('mma_graph', 'figure'),
              Input('mma_indicator1_dropdown', 'value'),
              Input('mma_indicator2_dropdown', 'value'),
              Input('mma_indicator3_dropdown', 'value'),
              Input('mma_indicator4_dropdown', 'value'),
              Input('mma_check1', 'value'),
              Input('mma_check2', 'value'),
              Input('mma_slider', 'value'))
def plot_mma_graph(ind1, ind2, ind3, ind4, ch1, ch2, year):

    df = tidy[tidy['Тип'].isin(['городской округ', 'муниципальный район', 'муниципальный округ',
                                 'городской округ (закрытое адм.-тер. образование)']) & tidy['Год'].eq(year)]

    logx = False
    logy = False
    if 'log' in ch1:
        logx = True
    if 'log' in ch2:
        logy = True

    df['Размер метки'] = df[ind3].pow(1/1.5)
    try:
        fig = px.scatter(df,
                         x=ind1,
                         y=ind2,
                         size='Размер метки',
                         color=ind4,
                         height=700,
                         log_x=logx,
                         log_y=logy,
                         hover_name='Территория',
                         size_max = 30,
                         trendline="ols", trendline_scope="overall", trendline_color_override="black",
                         # trendline_options=dict(log_x=logx, log_y=logx)
                         )
    except Exception:
        fig = px.scatter(df,
                         x=ind1,
                         y=ind2,
                         size='Размер метки',
                         color=ind4,
                         height=700,
                         log_x=logx,
                         log_y=logy,
                         hover_name='Территория',
                         size_max=30,
                         trendline="ols", trendline_scope="overall", trendline_color_override="black",
                         # trendline_options=dict(log_x=logx, log_y=logx)
                         )
    fig.layout.paper_bgcolor = globalbgcolor
    return fig



@app.callback(Output('migration_flows', 'figure'),
              Input('migration_dropdown', 'value'))
def migration_flows_map(vvalue):
    if vvalue == 'Самолеты':
        transp = transport[transport.transport == 'avia']
    elif vvalue == 'Поезда':
        transp = transport[transport.transport == 'train']
    else:
        transp = transport[transport.transport.isin(['avia','train'])]
    uniqtransp = np.unique(transp[['departure', 'arrival']].values)
    transport_cities = transport_cit[transport_cit['name'].isin(uniqtransp)]
    transport_cities['arr_dep'] = 0
    dd = dict(zip(transport_cities.name, transport_cities.id))
    flows = pd.merge(transp, transport_cities, left_on='departure', right_on='name', how='left')
    flows = pd.merge(flows, transport_cities, left_on='arrival', right_on='name', how='left')

    ColorDict = {'avia': 'red', 'train': 'blue', 'bus': 'green'}
    fig = go.Figure()
    fig.add_trace(go.Choropleth(
        geojson=rusmo, featureidkey="properties.id",
        locations=dff['ОКТМО'],
        z=dff['plotn'], hoverinfo='none',
        marker=dict(line_width=0.5, line_color='rgb(140,140,140)'),
        colorscale=[[0, 'rgb(0.8, 0.9, 0.8)'], [1, 'rgb(0.8, 0.9, 0.8)']], showscale=False
    ))
    for i in range(len(flows)):
        fig.add_trace(
            go.Scattergeo(
                lon=[flows['lng_x'][i], flows['lng_y'][i]],
                lat=[flows['lat_x'][i], flows['lat_y'][i]],
                mode='lines',
                line=dict(width=1, color=ColorDict[flows['transport'][i]]),
                opacity=float(flows['passengers'][i]) ** (1 / 1.5) / float(flows['passengers'].max()) ** (1 / 1.5)
            )
        )
    fig.add_trace(go.Scattergeo(
        lon=transport_cities['lng'],
        lat=transport_cities['lat'],
        text=transport_cities['name'],
        customdata=transport_cities['arr_dep'],
        hoverinfo='text',
        # hovertemplate = ' %{customdata} жителей',
        mode='markers',
        marker=dict(
            size=3,
            color='rgb(200, 40, 40)',
            line=dict(
                width=3,
                color='rgba(68, 68, 68, 0)'
            )
    )))

    fig.update_geos(projection_type="conic equal area",
                    projection_rotation_roll=0,
                    projection_rotation_lat=15,
                    projection_rotation_lon=120,
                    center_lat=64.1, # projection_parallels = [0, 100],
                    center_lon=94,
                    projection_scale=4.2,
                    resolution = 110,
                    showcountries=True)
    fig.update_layout(
        title_text='Перемещения людей в российской Арктике по данным Туту.ру в апреле 2019 г.<br>(красным цветом отмечены авиационные маршруты, синим - железнодорожные)',
        showlegend=False,
        height=700,
    )
    fig.layout.geo.bgcolor = globalbgcolor
    fig.layout.paper_bgcolor = globalbgcolor
    fig.layout.geo.landcolor = 'rgb(245, 245, 245)'
    return fig

@app.callback(Output('clustered_map_chart', 'figure'),
              Input('clustering_submit_button', 'n_clicks'),
              State('year_cluster_slider', 'value'),
              State('ncluster_cluster_slider', 'value'),
              State('cluster_indicator_dropdown', 'value'))
def clustered_map(n_clicks, year, n_clusters, indicators):
    if not indicators:
        raise PreventUpdate
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    scaler = StandardScaler()
    kmeans = KMeans(n_clusters=n_clusters)

    df = tidy[tidy['Уровень'].eq('муниципальное образование') & tidy['Год'].eq(year)][indicators + ['Территория', 'Год', 'ОКТМО']]
    df = pd.merge(df, coords, left_on='Территория', right_on='Территория', how='left')
    data = df[indicators]
    if df.isna().all().any():
        return px.scatter(title='Нет доступных данных для выбранной комбинации года и показателей.')
    data_no_na = imp.fit_transform(data)
    scaled_data = scaler.fit_transform(data_no_na)
    kmeans.fit(scaled_data)
    df['Кластер'] = [str(x) for x in kmeans.labels_]

    colors = {            '0': '#8dd3c7',
                          '1': '#ffffb3',
                          '2': '#bebada',
                          '3': '#fb8072',
                          '4': '#80b1d3',
                          '5': '#fdb462',
                          '6': '#b3de69',
                          '7': '#fccde5',
                          '8': '#d9d9d9',
                          '9': '#bc80bd'}


    df['Цвет'] = [colors[str(x)] for x in kmeans.labels_]

    fig = px.choropleth(df,
                        geojson = rusmo,
                        featureidkey = "properties.id",
                        locations = df['ОКТМО'],
                        color=[str(x) for x in kmeans.labels_],
                        labels={'color': 'Cluster'},
                        hover_data=indicators,
                        height=650,
                        title=f'Кластеры территорий - {year}. Число кластеров: {n_clusters}<br>Качество модели: {kmeans.inertia_:,.2f}',
                        #color_discrete_sequence=px.colors.qualitative.T10
                        color_discrete_map = colors
                    )
    fig.add_annotation(x=0.2, y=-0.15,
                       xref='paper', yref='paper',
                       text='Показатели:<br>' + "<br>".join(indicators),
                       showarrow=False)

    fig.add_scattergeo(
            lon = df['Долгота'],
            lat = df['Широта'],
            marker_color=df["Цвет"],
            marker_size = df['Размер'],
            marker_colorscale = px.colors.qualitative.T10,
            marker_line=dict(width=1,color='rgb(20, 20, 20)'),
            text = df['Территория'],
         #   marker_hover_data=df[indicators],
            customdata = df['Кластер'],
            hovertemplate = '%{text}<br>Номер кластера: %{customdata}<extra></extra>',
            name=""
    )


    fig.update_geos(projection_type="conic equal area", projection_rotation_roll=0,
                    projection_rotation_lat=15,
                    projection_rotation_lon=105,
                    center_lat=71,
                    center_lon=107,
                    projection_scale=6,
                    showcountries=True, countrywidth=0.5, coastlinewidth=0.5,
                    )
    fig.layout.geo.showcountries = True
    fig.layout.geo.landcolor = 'white'
    fig.layout.geo.bgcolor = globalbgcolor
    fig.layout.paper_bgcolor = globalbgcolor
    fig.layout.geo.countrycolor = 'gray'
    fig.layout.geo.coastlinecolor = 'gray'
    return fig


@app.callback(Output('country_page_contry_dropdown', 'value'),
              Input('location', 'pathname'))
def set_dropdown_values(pathname):
    if unquote(pathname[1:]) in areas:
        area = unquote(pathname[1:])
        return [area]


@app.callback(Output('country_heading', 'children'),
              Output('country_page_graph', 'figure'),
              Output('urban_table', 'children'),
              Output('age_pyramid', 'figure'),
              Output('edu_graph', 'figure'),
              Output('country_table', 'children'),
              Input('location', 'pathname'),
              Input('country_page_contry_dropdown', 'value'),
              Input('country_page_indicator_dropdown', 'value'))
def plot_country_charts(pathname, areas, indicator):
    if indicator == 'Численность населения (по переписям), человек' or \
            indicator == 'Плотность населения (по данным переписей), человек на 1 кв. км':
        years = [1939, 1959, 1970, 1979, 1989, 2002, 2010, 2021]
    else:
        years = [x for x in range(2010,2022)]

    profiles2 = profiles
    urban2 = urban
    if (not areas) or (not indicator):
        raise PreventUpdate
    if unquote(pathname[1:]) in areas:
        area = unquote(pathname[1:])
    df = tidy[tidy['Территория'].isin(areas) & tidy['Год'].isin(years)]
    fig = px.line(df,
                  x='Год',
                  y=indicator,
                  title='<b>' + indicator + '</b><br>' + ', '.join(areas),
                  color='Территория')
    fig.layout.paper_bgcolor = globalbgcolor

    # Таблица городов и пгт
    mos = urban2['МО'].drop_duplicates().tolist()
    regions = urban2['Регион'].drop_duplicates().tolist()
    if areas[0] in areas:
        if areas[0] in regions:
            table0 = urban2[urban2['Регион'] == areas[0]].iloc[:, 0:9].reset_index()
        elif areas[0] == 'Арктическая зона РФ':
            table0 = urban2.iloc[:, 0:9].reset_index()
        else:
            table0 = urban2[urban2['МО'] == areas[0]].iloc[:, 0:9].reset_index()
    if len(table0.index) > 0:
        table0 = dbc.Table.from_dataframe(table0.iloc[:, 1:10])
    else:
        table0 = html.Div()

    age_title = area
    x_F = agesex[area].to_numpy()[101:202].astype(int)
    x_M = agesex[area].to_numpy()[1:100].astype(int)
    x_M *= -1
    y_age = [i for i in range(101)]
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(y=y_age, x=x_M,
                          name='Мужчины',
                          customdata=-1 * x_M,
                          hovertemplate='В возрасте %{y} лет %{customdata} жителей',
                          marker=dict(color='blue', opacity=0.7),
                          orientation='h'))
    fig2.add_trace(go.Bar(y=y_age, x=x_F,
                          name='Женщины',
                          hovertemplate='В возрасте %{y} лет %{x} жителей',
                          marker=dict(color='red', opacity=0.7),
                          orientation='h'))
    fig2.update_layout(yaxis=go.layout.YAxis(
        title='Возраст, лет', showgrid=True,
        tickvals=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], ),
        xaxis=go.layout.XAxis(
            title='Численность населения, человек'),
        barmode='overlay',
        legend_orientation="h",
        legend=dict(x=.5, xanchor="center"),
        height=800, title_text=age_title + ". Состав населения по полу и возрасту, 2021 г.",
        bargap=0.1, paper_bgcolor=globalbgcolor
    )


    edu = education
    edu_cols = edu.columns


    if areas[0] is None:
        raise PreventUpdate
    fig3 = px.bar(edu[edu['Территория']==areas[0]].dropna(),
                 x=edu_cols,
                 y='Год',
                 barmode='stack',
                 height=400,
                 hover_name='Территория',
                 title=f'Образовательный состав населения {areas[0]}',
                 orientation='h')
    fig3.layout.legend.title = None
   # fig3.layout.legend.orientation = 'h'
   # fig3.layout.legend.x = 0.2
   # fig3.layout.legend.y = -0.15
    fig3.layout.xaxis.title = 'Доля от населения старше 15 лет, %'
    fig3.layout.paper_bgcolor = globalbgcolor
    fig3.layout.plot_bgcolor = globalbgcolor


    table1 = profiles2[profiles2['Территория'] == areas[0]].iloc[:, 8:31]
    table2 = profiles2[profiles2['Территория'] == areas[0]].iloc[:, 31:54]
    table3 = profiles2.iloc[0:1, 8:31]
    table2.columns = table1.columns
    table = pd.concat([table1, table2, table3]).T.reset_index()


    if table.shape[1] == 4:
        table.columns = ['Показатель' , 'Значение в ' + areas[0], 'Ранг', 'В целом по АЗРФ']
        table = dbc.Table.from_dataframe(table)
    else:
        table = html.Div()
    return 'Профиль ' + area , fig, table0, fig2, fig3, table

app.title = "Цифровой двойник населения Арктики. Дашборд"
if __name__ == '__main__':
    app.run_server(debug=False)
