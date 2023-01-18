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
import base64
import dash_cytoscape as cyto
import networkx as nx
import dash_daq as daq
import _datetime
from geopy.distance import geodesic as GD

globalbgcolor = '#F5FAFA'   # '#f7fbff'
np.warnings.filterwarnings('ignore')

app = dash.Dash(__name__, 
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0, maximum-scale=4, minimum-scale=0.5,'}],

              # 1   external_stylesheets=["assets/custom.css"])
                external_stylesheets=[dbc.themes.BOOTSTRAP])  # BOOTSTRAP COSMO PULSE ZEPHYR MATERIA LITERA
server = app.server

image_filename_1 = 'data/rsf_logo.png'
image_filename_2 = 'data/iespn_logo.png'
encoded_image_1 = base64.b64encode(open(image_filename_1, 'rb').read())
encoded_image_2 = base64.b64encode(open(image_filename_2, 'rb').read())

tidy = pd.read_csv('data/tidy.csv', delimiter = ';', low_memory=False)
indicators_info = pd.read_csv('data/indicators.csv', delimiter = ';')
profiles = pd.read_csv('data/profiles.csv', delimiter = ';', low_memory=False)
molist = pd.read_csv('data/molist.csv', delimiter = ';', low_memory=False)
stats = pd.read_csv('data/stats.csv', delimiter = ';', low_memory=False)
agesex = pd.read_csv('data/age_2021.csv', delimiter = ';', low_memory=False)
covid = pd.read_csv("data/covid_tidy_ma.csv", sep=';', low_memory=False)
covid_indicators_info = pd.read_csv('data/covid_indicators.csv', delimiter = ';')
transport = pd.read_csv("data/transport.csv", sep=',')
transport_cit = pd.read_csv("data/transport_cities.csv", sep=',')
urban = pd.read_csv('data/urban.csv', delimiter = ';')
education = pd.read_csv('data/education.csv', delimiter = ';')
coords = pd.read_csv('data/coords.csv', delimiter = ';')
mo_colors = pd.read_csv("data/mo_mcolor.csv", sep=';')
oktmo = pd.read_csv('data/arctic_oktmo.csv', sep=';')
science = pd.read_csv("data/science_tidy.csv", sep=';', low_memory=False)
higheredu = pd.read_csv("data/higheredu.csv", sep=';', low_memory=False)
mo_info = pd.read_csv("data/mo_info.csv", sep=';', low_memory=False)
settlements= pd.read_csv("data/settlements.csv", sep=';')
urban_t = pd.read_csv('data/urban_tidy.csv', delimiter = ';', low_memory=False)

areas = molist['Территория'].drop_duplicates().tolist()
with open('data/rusmo10_10.geojson', encoding='utf-8') as json_file:
    rusmo = json.load(json_file)


arcticmigrtable = pd.read_csv('data/arcticmigration.csv')
arcticmigrtable = arcticmigrtable[arcticmigrtable['migranty'] > 49]
migrnames = oktmo.migrname.tolist()
G = nx.DiGraph()
for i, row in arcticmigrtable.iterrows():
    G.add_edge(row[4], row[5], weight=row[3])
degree_values = [v for k, v in G.degree(weight='weight')]
in_degree_values = [v for k, v in G.in_degree(weight='weight')]
out_degree_values = [v for k, v in G.out_degree(weight='weight')]
cy = nx.cytoscape_data(G)
counter1 = 0
for n in cy["elements"]["nodes"]:
    for k, v in n.items():
        v["label"] = v.pop("value")
        v["size"] = degree_values[counter1]
        v["in_size"] = in_degree_values[counter1]
        v["out_size"] = out_degree_values[counter1]
        v["sq_size"] = degree_values[counter1] ** (1/2)
        counter1 = counter1 + 1
        if v["label"] in migrnames:
            v["color"] = "#1f78b4"
        else:
            v["color"] = "#e31a1c"
elements = cy["elements"]["nodes"] + cy["elements"]["edges"]
cyto_stylesheet=[
    {"selector": "node", "style": {
                                               "width": "mapData(sq_size, 0, 165, 3, 12)",
                                               "height": "mapData(sq_size, 0, 165, 3, 12)",
                                               "background-color": 'data(color)',
                                               "border-color": "#000000",
                                               'content': 'data(label)',
                                               'font-size': "mapData(sq_size, 0, 165, 4, 9)",
                                               "border-width": "0.2",
    }},
    {"selector": 'edge', "style": {"width": "mapData(weight, 0, 2000, 0.1, 1.0)",
                                               'target-arrow-color': '#b0b0b0', 'curve-style': 'bezier',
                                               'target-arrow-shape': 'triangle', 'arrow-scale': 0.1,
                                               'line-color': '#b0b0b0',
    }}
]


def make_empty_fig():
    fig = go.Figure()
    fig.layout.paper_bgcolor = globalbgcolor
    fig.layout.plot_bgcolor = globalbgcolor
    return fig


def multiline_indicator(indicator):
    final = []
    split = indicator.split()
    for i in range(0, len(split), 1):
        final.append(' '.join(split[i:i+1]))
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
                    dbc.Row([
                        dbc.Col([
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
                                html.Li('Последнее обновление: 10.01.2023')
                            ]),
                        ], xl=8, lg=7, md=6),
                        dbc.Col([
                            html.Br(),
                            html.Img(src='data:image/png;base64,{}'.format(encoded_image_1.decode()), width=125),
                            html.B("_____", style={'color': globalbgcolor}),
                            html.Img(src='data:image/png;base64,{}'.format(encoded_image_2.decode()), width=125),
                            html.Br(), html.Br(),
                        ], style={"textAlign": "right"}, xl=4, lg=5, md=6)
                    ])
                ], label='О проекте'),
                dbc.Tab([
                    dbc.Row([
                        dbc.Col([
                            html.Ul([
                                html.Br(),
                                html.Li(['Научные статьи о населении Арктики: ',
                                    html.A('http://vvfauzer.ru/index/arctic/0-18',
                                       href='http://vvfauzer.ru/index/arctic/0-18')
                                ])
                            ])
                        ], xl=8, lg=7, md=6),
                        dbc.Col([
                            html.Br(),
                            html.Img(src='data:image/png;base64,{}'.format(encoded_image_1.decode()), width=125),
                            html.B("_____", style={'color': globalbgcolor}),
                            html.Img(src='data:image/png;base64,{}'.format(encoded_image_2.decode()), width=125),
                            html.Br(), html.Br(),
                        ], style={"textAlign": "right"}, xl=4, lg=5, md=6)
                    ])
                ], label='Полезные ссылки')
            ]),
        ], lg=10)
    ])
], style={ 'backgroundColor': globalbgcolor
     })
])

area_dashboard = html.Div([
    dbc.Row([
        dbc.Col(lg=2),
        dbc.Col([
    html.Br(),
    html.H1('Цифровой двойник населения Арктики', style={'textAlign': 'center'}),
    html.Br(),
        html.H2(id='country_heading'),
        dcc.Markdown(id='area_info', style={'backgroundColor': globalbgcolor} ),
        dbc.Row([
            dbc.Col(dcc.Graph(id='country_page_graph'))
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Label('Выберите показатель:'),
                dcc.Dropdown(id='country_page_indicator_dropdown',
                             placeholder='Выберите показатель',
                             value='Численность населения (по переписям), человек',
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
        ], lg=8)
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
                                 value='Специальный коэффициент рождаемости',
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
                                       included=False, dots = True,
                                       value=2021,
                                #       tooltip={"placement": "top", "always_visible": True},
                                       marks={year: {'label': str(year),
                                                     'style': {'color': 'black', 'fontSize': 14}}
                                              for year in range(2010, 2023, 2)}),
                        ], lg=7),
                        dbc.Col([
                            dbc.Label('Цветовая шкала:'),
                            dcc.Dropdown(id='indicator_map_color_dropdown',
                                         value='Спектральная',
                                         options=[{'label': indicator,
                                         'value': indicator}
                                         for indicator in ['Спектральная', 'От красного к синему',
                                                           'От красного к зеленому','Для ч/б печати' , 'При нарушении цветовосприятия']]),
                        ], lg=3),
                        dbc.Col([
                            daq.BooleanSwitch(id='indicator_map_inverter', on=False, label="Обратная шкала")
                        ], lg=2),
                    ]),
                    dcc.Graph(id='indicator_map_chart'),
                    dcc.Markdown(id='indicator_map_details_md',
                                style={'backgroundColor': globalbgcolor}
                         ),
                    dcc.Graph(id='indicator_year_barchart',
                             figure=make_empty_fig())
                ], label='Показатели'),

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
                            daq.BooleanSwitch(id='mma_check1', on=False, label="Логарифмическая шкала")
                        ], lg=4),
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label('Показатель по оси Y:'),
                            dcc.Dropdown(id='mma_indicator2_dropdown', optionHeight=40,
                                         value='Общая демографическая нагрузка',
                                         options=[{'label': indicator, 'value': indicator}
                                                  for indicator in tidy.columns[9:65]]),
                        ], lg=8),
                        dbc.Col([
                            daq.BooleanSwitch(id='mma_check2', on=False, label="Логарифмическая шкала")
                        ], lg=4),
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label('Размер метки:'),
                            dcc.Dropdown(id='mma_indicator3_dropdown',optionHeight=40,
                                            value='Среднегодовая численность населения, человек',
                                            options=[{'label': indicator, 'value': indicator}
                                                    for indicator in ['Среднегодовая численность населения, человек',
                                                                      'Площадь территории, кв. км',
                                                                      'Отгружено товаров собственного производства, выполнено работ и услуг, млн рублей']]),
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
                                                         'style': {'color': 'black', 'fontSize': 14}}
                                                  for year in [x for x in range(2010, 2022)]}),
                                ], lg=12),
                    ]),
                    html.Br(),
                    dcc.Graph(id='mma_graph',figure=make_empty_fig()),
                ], label='Многомерный анализ'),

                dbc.Tab([
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label('Выберите год:'),
                            dcc.Slider(id='year_cluster_slider',
                                    min=2010, max=2021, step=1, included=False,
                                    value=2021, dots = True,
                                       marks={year: {'label': str(year),
                                                     'style': {'color': 'black', 'fontSize': 14}}
                                              for year in range(2010, 2023, 2)}),
                        ], lg=7, md=12),
                        dbc.Col([
                            dbc.Label('Выберите число кластеров:'),
                            dcc.Slider(id='ncluster_cluster_slider',
                                    min=2, max=10, step=1, included=False,
                                    value=4,
                                    marks={n:  {'label': str(n),
                                                'style': {'color': 'black', 'fontSize': 14}}
                                           for n in range(2, 11)}),
                        ], lg=5, md=12)
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label('Выберите показатели:'),
                            dcc.Dropdown(id='cluster_indicator_dropdown',optionHeight=40,
                                        multi=True,
                                        value=['Среднемесячная заработная плата работников организаций, рублей'],
                                        options=[{'label': indicator, 'value': indicator}
                                                for indicator in tidy.columns[9:65]]),
                            html.Br(),
                        ], lg=7),
                        dbc.Col([            
                            dbc.Label(''),html.Br(),
                            dbc.Button("Отправить", id='clustering_submit_button'),
                        ]),
                    ]),
                    dcc.Loading([
                        dcc.Graph(id='clustered_map_chart'),
                        html.H4('Средние значения показателей по кластерам'),
                        html.Div(id='cluster_table'),
                    ]),
                    html.Li(['Кластеризация по методу k-средних с использованием стандартного масштабирования данных. ',
                            'Для оценки качества модели рассчитывается сумма квадратов расстояний от исходных точек ',
                            'до ближайших к ним кластеров (функция inertia пакета scikit-learn). Чем ниже значение, тем лучше. ']),
                ], label='Кластеризация'),

                dbc.Tab([
                    html.Br(),
                    dbc.Label('Выберите показатель:'),
                    dcc.Dropdown(id='ind_forecast_dropdown',
                                 value='Общий коэффициент смертности, промилле',
                                 options=[{'label': indicator,
                                           'value': indicator}
                                          for indicator in tidy.columns[9:65]]),
                    html.Br(),
                    dbc.Label('Выберите территорию:'),
                    dcc.Dropdown(id='area_forecast_dropdown',
                                 value='Арктическая зона РФ',
                                 options=[{'label': area,
                                           'value': area}
                                          for area in areas]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label('Выберите годы учитываемых значений:'),
                            dcc.RangeSlider(id='forecast_slider',
                                            min=2010, max=2021, step=1,
                                            value=[2010, 2019], dots=True,
                                            marks={year: {'label': str(year),
                                                          'style': {'color': 'black', 'fontSize': 14}}
                                                   for year in range(2010, 2023, 2)}),
                        ], lg=7, md=12),
                        dbc.Col([
                            dbc.Label('Выберите степень полинома:'),
                            dcc.Slider(id='forecast_degree',
                                       min=1, max=10, step=1, included=False,
                                       value=1,
                                       marks={n: {'label': str(n),
                                                  'style': {'color': 'black', 'fontSize': 14}}
                                              for n in range(1, 11)}),
                        ], lg=5, md=12)
                    ]),
                    dcc.Graph(id='forecast_graph'),
                ], label='Прогноз'),

                dbc.Tab([
                    html.Br(),
                    dbc.Label('Выберите карту'),
                    dcc.Dropdown(id='settlement_dropdown', optionHeight=40,
                                 value='Карта расселения',
                                 options=[{'label': indicator, 'value': indicator}
                                          for indicator in ['Карта расселения', 'Крупнейшая национальность',
                                                            'Вторая национальность', 'Анимация городского расселения']]),
                    dcc.Graph(id='settlement_map'),
                    html.Br(), html.Br(),

                    html.H3('Анализ центров расселения'),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label('Выберите населенный пункт'),
                            dcc.Dropdown(id='settlement_dropdown2', optionHeight=40,
                                                             value='г. Архангельск',
                                                             options=[{'label': indicator, 'value': indicator}
                                                                      for indicator in settlements['Населенный пункт']]),
                        ]),
                        dbc.Col([
                            dbc.Label('Выберите радиус в километрах'),
                            dcc.Dropdown(id='settlement_dropdown3', optionHeight=40,
                                             value='50',
                                             options=[{'label': indicator, 'value': indicator}
                                                      for indicator in [5, 10, 20, 50, 100, 150]]),
                        ]),
                    ]),
                    html.Br(),
                    dcc.Markdown(id='settlement_details', style={'backgroundColor': globalbgcolor} ),
                    html.Br(),
                    html.Div(id='settlement_table'),

                ], label='Расселение'),

                dbc.Tab([
                    html.Br(),
                    html.H4('Миграционные потоки в Арктике по данным проекта "Виртуальное население России"'),
                    html.B('Наведите курсор на узел сети или поток. '),
                    html.Br(),
                    cyto.Cytoscape(
                        id="cytoscape_migration",
                        zoom=1.7,
                        elements=elements,
                        style={"width": "100%", "height": "650px"},
                        layout={"name": "cose", "fit": False},  # "preset" to use the pos coords
                        stylesheet=cyto_stylesheet,
                    ),

                    html.B(id='cytoscape-mouseoverNodeData-output'),
                    html.Br(),
                    html.B(id='cytoscape-mouseoverEdgeData-output'),
                    html.Br(),

                    html.Li(['Составлено по данным профилей социальной сети "ВКонтакте". ',
                            'Учитывается только последняя смена места жительства ',
                            'Синим цветом отмечены арктические города, красным - остальные. ',
                            'На сайте представлена упрощенная модель, содержащая только потоки от 50 человек. ',
                            'Наведите на узел или поток для получения подробной информации.']),
                    html.Li(['Полный набор данных доступен на сайте проекта "' ,
                        html.A((html.B('Виртуальное население России')),
                                href='https://story.tutu.ru/dataset-tutu-ru-i-dannye-modeli-open-data-science/'), '".'
                            ]),
                    html.Li([
                        'Пример изучения перемещений населения методами сетевого анализа представлен в статье "',
                        html.B(html.A('Цифровые следы населения как источник данных о миграционных потоках в российской Арктике',
                                href='https://www.avsci.ru/p/1_25.pdf')), '".'
                    ])
                ], label='Миграция'),

                dbc.Tab([
                    html.Br(),
                    dbc.Label('Выберите виды транспорта:'),
                    dcc.Dropdown(id='migration_dropdown',
                                 value='Самолеты и поезда',
                                 options=[{'label': param,
                                 'value': param}
                                 for param in ['Самолеты и поезда','Самолеты','Поезда']]),

                    dcc.Graph(id='migration_flows'),

                    html.Li([
                        html.A(('Полный набор данных на сайте ', html.B('Туту.ру'), ' и его описание'),
                                href='https://story.tutu.ru/dataset-tutu-ru-i-dannye-modeli-open-data-science/')
                            ]),
                    html.Li([
                        'Пример изучения перемещений населения методами сетевого анализа представлен в статье "',
                        html.B(html.A('Цифровые следы населения как источник данных о миграционных потоках в российской Арктике',
                                href='https://www.avsci.ru/p/1_25.pdf')), '"'
                    ])
                ], label='Транспорт'),

                dbc.Tab([
                    html.Br(),
                    html.H4('Комплексный балл публикационной результативности по организациям'),
                    dbc.Label('Выберите направление науки:'),
                    dcc.Dropdown(id='science_dropdown',
                                 value='Все направления',
                                 options=[{'label': param,
                                 'value': param}
                                 for param in ['Все направления','Математика','Компьютерные и информационные науки',
                                               'Физические науки','Химические науки','Науки о Земле', 'Биологические науки',
                                               'Технические науки', 'Медицинские науки', 'Сельскохозяйственные науки',
                                               'Общественные науки', 'Гуманитарные науки']]),
                    dcc.Loading([
                        dcc.Graph(id='science_barchart')
                    ]),
                    html.Br(),
                    html.H4('Приведенный контингент студентов вузов по организациям'),
                    dbc.Label('Выберите отрасль наук:'),
                    dcc.Dropdown(id='higheredu_dropdown',
                                 value='Все отрасли наук',
                                 options=[{'label': indicator, 'value': indicator}
                                          for indicator in higheredu.columns[2:11]]),
                    dcc.Loading([
                        dcc.Graph(id='higheredu_barchart')

                    ]),
                    html.Li(['Комплексный балл публикационной результативности (КБПР) характеризует публикационную ',
                             'результативность научного сотрудника и рассчитывается с учетом квартильности и категории ',
                             'научных публикаций методом фракционного счета (разделением вклада авторов в научный ',
                             'результат в случае, если публикация подготовлена несколькими авторами и из разных организаций). ',
                             'Источник данных: Научная электронная библиотека eLIBRARY.RU.',
                    ]),
                    html.Li(['Приведенный контингент студентов рассчитывается по формуле: а + (b * 0,25) + ((c+d) х 0,1), ',
                             'где: а - численность студентов очной формы обучения; b - численность студентов очно-заочной ',
                             '(вечерней) формы обучения; с - численность студентов заочной формы обучения, ',
                             'd - численность студентов экстерната). Оценки взяты из иформационно-аналитических материалов ',
                             'по результатам проведения мониторинга деятельности образовательных организаций высшего образования.',
                    ]),
                ], label='Наука и образование'),

                dbc.Tab([
                    html.Br(),
                    dbc.Label('Выберите показатель:'),
                    dcc.Dropdown(id='covid_indicator_dropdown',
                             placeholder='Выберите показатель',
                             value='Заражений за день, на 1 млн человек',
                             options=[{'label': indicator, 'value': indicator}
                                     for indicator in covid_indicators_info['Показатель'].drop_duplicates().tolist()]),
                    html.Br(),
                    dcc.Graph(id='covid_graph'),
                    dcc.Markdown(id='covid_details', style={'backgroundColor': globalbgcolor} ),
                    html.Br(),
                    html.Li([
                        'Пример изучения пандемии с помощью временных рядов показателей представлен в статье "',
                        html.B(html.A('Влияние пандемии на демографические процессы в Российской Арктике',
                                href='https://www.avsci.ru/p/1_21.pdf')), '"'
                    ])
                ], label='Пандемия'),
            ]),
        ], lg=8, md=12)
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

@app.callback(Output('cytoscape-mouseoverNodeData-output', 'children'),
              Input('cytoscape_migration', 'mouseoverNodeData'))
def displayTapNodeData(data):
    if data:
        return "Выбран узел: " + data['label'] + ". Прибывших: " + str(data['in_size']) + ". Выбывших: " + str(data['out_size']) + "."

@app.callback(Output('cytoscape-mouseoverEdgeData-output', 'children'),
              Input('cytoscape_migration', 'mouseoverEdgeData'))
def displayTapNodeData(data):
    if data:
        return "Выбран поток: " + data['source'] + " - " + data['target'] + ". Число перемещений: " + str(data['weight']) + "."


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
              Input('indicator_map_inverter','on'))
def display_generic_map_chart(indicator, year, mapcolor, invert):
    color_dict = {
        "Спектральная": "spectral",
        "От красного к синему": "RdBu",
        "От красного к зеленому": "RdYlGn",
        "Для ч/б печати": "Viridis",
        "При нарушении цветовосприятия": "Cividis",
    }

    imapcolor = color_dict[mapcolor]
    if invert is True:
        imapcolor = imapcolor + '_r'

    if indicator is None:
        raise PreventUpdate

    dat = tidy[tidy['Тип'].isin(['городской округ', 'муниципальный район', 'муниципальный округ',
                                'городской округ (закрытое адм.-тер. образование)'])  & tidy['Год'].eq(year)]

    df = pd.merge(dat, coords, left_on='Территория',right_on='Территория',how='left')
    df.loc[np.isnan(df[indicator]), 'Размер'] = 0

    fig = go.Figure()
    fig.add_trace(go.Choropleth(
        geojson=rusmo, featureidkey="properties.id",
        locations=df['ОКТМО'],
        z=df[indicator], hoverinfo='text',
        marker=dict(line_width=0.5, line_color='rgb(140,140,140)'),
        colorscale=imapcolor, showscale=True,
        colorbar_title = multiline_indicator(indicator) + ",<br>" + str(year) + " г.",
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
                opacity = 1,
                line=dict(width=1,color='rgb(20, 20, 20)'),
            ),
            hovertemplate='%{text}<br>Значение: %{customdata}<extra></extra>',
        ))
    except Exception:
        aaa = 1

    fig.update_geos(projection_type="conic equal area", projection_rotation_roll=0,
                    projection_rotation_lat=15,
                    projection_rotation_lon=105,
                    center_lat=70.7,
                    center_lon=107,
                    projection_scale=6.2, showcoastlines = False,
                    showcountries=False, countrywidth = 0.5, coastlinewidth = 0.5,
                    )
    fig.update_layout(
        height=650,  margin=dict(l=20, r=20, t=25, b=25),
    )
    fig.layout.geo.bgcolor = globalbgcolor
    fig.layout.paper_bgcolor = globalbgcolor
    fig.layout.geo.landcolor = 'rgb(240, 240, 240)'


    
    series_df = indicators_info[indicators_info['Показатель'].eq(indicator)]
    if series_df.empty:
        markdown = "Нет данных по данному показателю"
    else:
        limitations = series_df['Ограничения и комментарий'].fillna('отсутствуют').str.replace('\n\n', ' ').values[0]

        markdown = f"""
        ## {series_df['Показатель'].values[0]}  
        
        {series_df['Описание'].values[0]}  
        
        * **Группа показателей:** {series_df['Группа показателей'].fillna('count').values[0]}
        * **Единица измерения:** {series_df['Единица измерения'].fillna('count').values[0]}
        * **Периодичность:** {series_df['Период'].fillna('N/A').values[0]}
        * **Форма отчетности:** {series_df['Форма отчетности'].fillna('N/A').values[0]}
        * **Источник:** {series_df['Источник'].values[0]}
        
        #### Ограничения и комментарии:  
        
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
    fig2.update_layout(margin=dict(l=20, r=20, t=25, b=25))
    fig2.layout.paper_bgcolor = globalbgcolor



    return fig, markdown, fig2


@app.callback(Output('mma_graph', 'figure'),
              Input('mma_indicator1_dropdown', 'value'),
              Input('mma_indicator2_dropdown', 'value'),
              Input('mma_indicator3_dropdown', 'value'),
              Input('mma_indicator4_dropdown', 'value'),
              Input('mma_check1', 'on'),
              Input('mma_check2', 'on'),
              Input('mma_slider', 'value'))
def plot_mma_graph(ind1, ind2, ind3, ind4, ch1, ch2, year):

    df = tidy[tidy['Тип'].isin(['городской округ', 'муниципальный район', 'муниципальный округ',
                                 'городской округ (закрытое адм.-тер. образование)']) & tidy['Год'].eq(year)]

    logx = False
    logy = False
    if ch1 == True:
        logx = True
    if ch2 == True:
        logy = True

    df.loc[:, 'Размер метки'] = df[ind3].apply(lambda x: x ** (1/1.5))
    try:
        fig = px.scatter(df,
                         x=ind1,
                         y=ind2,
                         size='Размер метки',
                         color=ind4,
                         height=650,
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
                         height=650,
                         log_x=logx,
                         log_y=logy,
                         hover_name='Территория',
                         size_max=30,
                         trendline="ols", trendline_scope="overall", trendline_color_override="black",
                         # trendline_options=dict(log_x=logx, log_y=logx)
                         )
    fig.layout.paper_bgcolor = globalbgcolor
    fig.update_layout(margin=dict(l=20, r=20, t=25, b=25))
    return fig


@app.callback(Output('settlement_map', 'figure'),
              Output('settlement_details', 'children'),
              Output('settlement_table', 'children'),
              Input('settlement_dropdown', 'value'),
              Input('settlement_dropdown2', 'value'),
              Input('settlement_dropdown3', 'value'))
def settlement_map_plot(value, np, radius):
    if value == 'Карта расселения':
        settl = settlements
        colors = {'город': '#e31a1c', 'пгт': '#1f78b4', 'поселок': '#ffff99', 'сельский нп': '#a6cee3',
                  'село': '#33a02c', 'деревня': '#b2df8a', 'станция': '#fdbf6f'}
        settl['color'] = [colors[str(x)] for x in settl['typ']]
        settl['custom'] = 'Тип: ' + settl['typ'] + '<br>Население в 2020 г.: ' + settl['Население, человек'].astype(str) + '<br>(по данным ИНИД)'
        df = mo_colors
        fig = go.Figure()
        fig.add_trace(go.Choropleth(
            geojson=rusmo, featureidkey="properties.id",
            locations=df['ОКТМО'],
            z=df['plot'], hoverinfo='text',
            colorscale=[[0, 'rgb(0.8, 0.9, 0.8)'], [1, 'rgb(0.8, 0.9, 0.8)']]

        ))
        fig.update_traces(showscale=False)
        fig.update_traces(marker_line_width=0.35)
        fig.add_trace(go.Scattergeo(
            lon=settl['lon'],
            lat=settl['lat'],
            mode='markers',
            marker_size=settl['Население, человек'] ** (1 / 5) + 2,
            marker_color=settl['color'],
            text=settl['settlement'],
            customdata=settl['custom'],
            marker=dict(
                opacity=1,
                line=dict(width=0.2, color='rgb(20, 20, 20)'),
            ),
            hovertemplate='%{text}<br>%{customdata}<extra></extra>',
        ))


    elif value == 'Анимация городского расселения':
        fig = px.scatter_geo(urban_t, lon='Долгота', lat='Широта', size=urban_t["Численность населения"] ** (1 / 1.7)+2,
                             animation_frame='Год', color='Тип', hover_name='Название',
                             custom_data=['Численность населения'], hover_data=['Численность населения']
                             )

    else:
        mo_info2 = mo_info[mo_info['Тип'].isin(['городской округ', 'муниципальный район', 'муниципальный округ',
                                                'городской округ (закрытое административно-территориальное образование)'])]
        colors = {'русские': '#8dd3c7', 'ханты': '#ffffb3', 'ненцы': '#bebada', 'чукчи': '#fb8072', 'якуты': '#80b1d3',
                  'долганы': '#fdb462', 'эвенки': '#b3de69', 'эвены': '#fccde5', 'украинцы': '#b15928', 'коми': '#bc80bd',
                  'татары': '#1f78b4', 'селькупы': '#33a02c', 'эскимосы': '#e31a1c', 'кеты': '#ff7f00', 'карелы': '#6a3d9a',
                  'белорусы': '#999999'}



        mo_info2 = pd.merge(mo_info2, coords, left_on='Территория', right_on='Территория', how='left')

    #    mo_info2['Цвет'] = 0
    #    for index, row in mo_info2.iterrows():
    #        row['Цвет'] = colors[row[value]]

        mo_info2['Цвет'] =mo_info2.apply(lambda x: colors[x[value]], axis=1)

        fig = px.choropleth(mo_info2,
                            geojson=rusmo,
                            featureidkey="properties.id",
                            locations=mo_info2['ОКТМО'],
                            color=mo_info2[value],
                            color_discrete_map=colors,
                            hover_data=['Территория'],
                            )

        fig.add_scattergeo(
            lon=mo_info2['Долгота'],
            lat=mo_info2['Широта'],
            marker_color=mo_info2["Цвет"],
            marker_opacity=1,
            marker_size=mo_info2['Размер'],
            marker_colorscale=px.colors.qualitative.T10,
            marker_line=dict(width=1, color='rgb(20, 20, 20)'),
            text=mo_info2['Территория'],
            customdata=mo_info2[value],
            hovertemplate='%{text}<br>Национальность: %{customdata}<extra></extra>',
            name=""
        )

    fig.update_geos(projection_type="conic equal area", projection_rotation_roll=0,
                    projection_rotation_lat=15,
                    projection_rotation_lon=105,
                    center_lat=70.7,
                    center_lon=107,
                    projection_scale=6.2, showcoastlines=False,
                    showcountries=False, countrywidth=0.5, coastlinewidth=0.5,
                    )
    fig.update_layout(height=650)
    fig.layout.geo.bgcolor = globalbgcolor
    fig.layout.paper_bgcolor = globalbgcolor
    fig.layout.geo.landcolor = 'rgb(240, 240, 240)'
    fig.update_layout(margin=dict(l=20, r=20, t=25, b=25))

    lat0 = settlements[settlements['Населенный пункт'].eq(np)].iloc[0]['lat']
    lon0 = settlements[settlements['Населенный пункт'].eq(np)].iloc[0]['lon']
    settl = settlements
    radius_list = []
    for i, row in settl.iterrows():
        radius_list.append(round(GD((lat0, lon0), (row['lat'], row['lon'])).km, 2))
    settl['Расстояние, км'] = radius_list

    settl_table = settl[settl['Расстояние, км'] <= float(radius)].iloc[:, [0, 6, 18]].reset_index()
    settl_table["№"] = [x for x in range(1, settl_table.shape[0]+1)]

    if len(settl_table.index) > 0:
        table0 = dbc.Table.from_dataframe(settl_table.iloc[:, [4,1,2,3]], striped=True, bordered=True, hover=True)
    else:
        table0 = html.Div()
    summa = settl_table['Население, человек'].sum()

    markdown = f"""
            ** Население в радиусе {radius} км от {np} составляет {summa} чел. (по данным ИНИД на 2020 г.): **
            """

    return fig, markdown, table0


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
    dd = dict(zip(transport_cities.name, transport_cities.id))
    flows = pd.merge(transp, transport_cities, left_on='departure', right_on='name', how='left')
    flows = pd.merge(flows, transport_cities, left_on='arrival', right_on='name', how='left')

    transport_cities['arr'] = 0
    transport_cities['dep'] = 0
    for index, row in transp.iterrows():
        transport_cities.loc[dd[row['departure']], 'dep'] += row['passengers']
        transport_cities.loc[dd[row['arrival']], 'arr'] += row['passengers']
    transport_cities['size'] = (transport_cities['arr'] + transport_cities['dep']) ** (1/6) + 2


    transport_cities['custom'] = 'Выбывших: ' + transport_cities['dep'].astype(str) + '<br>Прибывших: ' + transport_cities['arr'].astype(str)

    ColorDict = {'avia': 'red', 'train': 'blue', 'bus': 'green'}
    fig = go.Figure()
    fig.add_trace(go.Choropleth(
        geojson=rusmo, featureidkey="properties.id",
        locations=mo_colors['ОКТМО'],
        z=mo_colors['plotn'], hoverinfo='none',
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
        customdata=transport_cities['custom'],
        hovertemplate='%{text}<br>%{customdata}<extra></extra>',
        hoverinfo='text',
        mode='markers',
        marker=dict(
            size=transport_cities['size'],
            color='rgb(200, 0, 100)',
            line=dict(
                width=3,
                color='rgba(68, 68, 68, 0)'
            )
    )))

    fig.update_geos(projection_type="conic equal area",
                    projection_rotation_roll=0,
                    projection_rotation_lat=10,
                    projection_rotation_lon=105,
                    center_lat=62, # projection_parallels = [0, 100],
                    center_lon=100,
                    projection_scale=4.2,
                    resolution = 110,
                    showcountries=True)
    fig.update_layout(
        title_text='Перемещения людей в российской Арктике по данным Туту.ру в апреле 2019 г.<br>(красным цветом отмечены авиационные маршруты, синим - железнодорожные)',
        showlegend=False,
        height=650,
    )
    fig.layout.geo.bgcolor = globalbgcolor
    fig.layout.paper_bgcolor = globalbgcolor
    fig.layout.geo.landcolor = 'rgb(245, 245, 245)'
    fig.update_layout(margin=dict(l=20, r=20, t=65, b=35))
    return fig

@app.callback(Output('science_barchart', 'figure'),
              Input('science_dropdown', 'value'))
def science_barchart_plot(indicator):
    fig = px.bar(science, x="Год", y=indicator, color="Организация", height=600)
    fig.update_layout(
        legend=dict(orientation="h", y=-0.2),
        xaxis=dict(tickmode='linear'),
        yaxis=dict(title='КБПР (' + indicator + ')')
    )
    fig.update_layout(margin=dict(l=20, r=20, t=30, b=25))
    fig.layout.paper_bgcolor = globalbgcolor
    fig.layout.plot_bgcolor = globalbgcolor
    return fig

@app.callback(Output('higheredu_barchart', 'figure'),
              Input('higheredu_dropdown', 'value'))
def higheredu_barchart_plot(indicator):
    fig = px.bar(higheredu, x="Год", y=indicator, color="Организация", height=600)
    fig.update_layout(
        legend=dict(orientation="h", y=-0.2),
        xaxis=dict(tickmode='linear'),
        yaxis = dict(title='Студентов (' + indicator + ')')
    )
    fig.update_layout(margin=dict(l=20, r=20, t=30, b=35))
    fig.update_yaxes(tickformat="none")
    fig.layout.paper_bgcolor = globalbgcolor
    fig.layout.plot_bgcolor = globalbgcolor
    return fig


@app.callback(Output('covid_graph', 'figure'),
              Output('covid_details', 'children'),
              Input('covid_indicator_dropdown', 'value'))
def covil_graph_plot(indicator):
    if indicator == 'Индекс самоизоляции':
        df = covid[covid['Регион'].isin(['Города России', 'Арктические города', 'Мурманск', 'Апатиты', 'Североморск',
                                         'Архангельск', 'Северодвинск', 'Воркута', 'Новый Уренгой', 'Ноябрьск',
                                         'Норильск'])]
    else:
        df = covid[covid['Регион'].isin(['Россия', 'Арктические регионы', 'Архангельская обл.', 'Мурманская обл.',
                                         'Ненецкий АО', 'Чукотский АО', 'Ямало-Ненецкий АО'])]
    try:
        fig = px.line(df, x='Дата', y=indicator, color='Регион')
    except Exception:
        fig = px.line(df, x='Дата', y=indicator, color='Регион')

    fig.update_layout(height=650)
    fig.update_xaxes(tickformat='%d.%m<br>%Y')
    fig.layout.paper_bgcolor = globalbgcolor
    fig.update_layout(margin=dict(l=20, r=20, t=30, b=25))

    series_df = covid_indicators_info[covid_indicators_info['Показатель'].eq(indicator)]
    if series_df.empty:
        markdown = "Нет данных по данному показателю"
    else:
        limitations = series_df['Ограничения и комментарий'].fillna('отсутствуют').str.replace('\n\n', ' ').values[0]

        markdown = f"""
            ## {series_df['Показатель'].values[0]}  

            {series_df['Описание'].values[0]}  

            * **Группа показателей:** {series_df['Группа показателей'].fillna('count').values[0]}
            * **Единица измерения:** {series_df['Единица измерения'].fillna('count').values[0]}
            * **Периодичность:** {series_df['Период'].fillna('N/A').values[0]}
            * **Источник:** {series_df['Источник'].values[0]}

            #### Ограничения и комментарии:  

            {limitations}  

            """

    return fig, markdown


@app.callback(Output('clustered_map_chart', 'figure'),
              Output('cluster_table', 'children'),
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
    labels = kmeans.labels_ + 1
    df['Номер кластера'] = [str(x) for x in labels]

    colors = { '1': '#8dd3c7', '2': '#ffffb3', '3': '#bebada', '4': '#fb8072', '5': '#80b1d3',
               '6': '#fdb462', '7': '#b3de69', '8': '#fccde5', '9': '#d9d9d9', '10': '#bc80bd'}


    df['Цвет'] = [colors[str(x)] for x in labels]

    fig = px.choropleth(df,
                        geojson = rusmo,
                        featureidkey = "properties.id",
                        locations = df['ОКТМО'],
                        color=[str(x) for x in labels],
                        labels={'color': 'Номер кластера'},
                        hover_data=indicators, hover_name='Территория',
                        height=650,
                        title=f'Кластеры территорий, {year}. Число кластеров: {n_clusters}. Качество модели: {kmeans.inertia_:,.2f}',
                        #color_discrete_sequence=px.colors.qualitative.T10
                        color_discrete_map = colors
                    )
    fig.add_annotation(x=0.2, y=-0.15,
                       xref='paper', yref='paper',
                       text='Показатели:<br>' + "<br>".join(indicators),
                       showarrow=False)
    fig.update_traces(marker_line_color='rgb(140,140,140)')


    fig.add_scattergeo(
            lon = df['Долгота'],
            lat = df['Широта'],
            marker_color=df["Цвет"],
            marker_opacity=1,
            marker_size = df['Размер'],
            marker_colorscale = px.colors.qualitative.T10,
            marker_line=dict(width=1,color='rgb(20, 20, 20)'),
            text = df['Территория'],
         #   marker_hover_data=df[indicators],
            customdata = df['Номер кластера'],
            hovertemplate = '%{text}<br>Номер кластера: %{customdata}<extra></extra>',
            name=""
    )


    fig.update_geos(projection_type="conic equal area", projection_rotation_roll=0,
                    projection_rotation_lat=15,
                    projection_rotation_lon=105,
                    center_lat=71,
                    center_lon=107,
                    projection_scale=6,
                    showcoastlines=False,
                    showcountries=False,
                    countrywidth=0.5,
                    coastlinewidth=0.5,
                    )

    fig.update_layout(margin=dict(l=20, r=20, t=35, b=35))
    fig.layout.geo.bgcolor = globalbgcolor
    fig.layout.paper_bgcolor = globalbgcolor
    fig.layout.geo.landcolor = 'rgb(240, 240, 240)'

    table00 = df.groupby(['Номер кластера'],as_index=False).mean().iloc[:, 0:len(indicators)+1].round(1)
    if len(table00.index) > 0:
        cluster_members = list()
        for c in range(1, n_clusters+1):
            cluster_members.append(len(df[df['Номер кластера'] == str(c)][:]))
        cluster_members
        table00.insert(1, "Количество территорий", cluster_members)
        table0 = dbc.Table.from_dataframe(table00)
    else:
        table0 = html.Div()


    return fig, table0


@app.callback(Output('country_page_contry_dropdown', 'value'),
              Input('location', 'pathname'))
def set_dropdown_values(pathname):
    if unquote(pathname[1:]) in areas:
        area = unquote(pathname[1:])
        return [area]

@app.callback(Output('forecast_graph', 'figure'),
              Input('ind_forecast_dropdown', 'value'),
              Input('area_forecast_dropdown', 'value'),
              Input('forecast_slider', 'value'),
              Input('forecast_degree', 'value'))
def plot_forecast(indicator, area, years, degree):
    df = tidy[tidy['Территория'].eq(area) & tidy['Год'].isin([x for x in range(years[0], years[1]+1)])][['Год',indicator]].dropna()
    df0 = tidy[tidy['Территория'].eq(area) & tidy['Год'].isin([x for x in range(2010, 2022)])][['Год', indicator]].dropna()
    x = df['Год']
    y = df[indicator]
    x0 = df0['Год']
    y0 = df0[indicator]
    z = np.polyfit(x, y, degree)
    f = np.poly1d(z)
    x_new = np.linspace(2010, 2030, 21)
    y_new = f(x_new)

    trace1 = go.Scatter(
        x=x0, y=y0, mode='markers', name='Все точки', marker=dict(size=10)
    )
    trace2 = go.Scatter(
        x=x, y=y, mode='markers', name='Учитываются<br>в прогнозе', marker=dict(size=10)
    )
    trace3 = go.Scatter(
        x=x_new, y=y_new, mode='lines', name='Прогноз'
    )
    data = [trace1, trace2, trace3]
    fig = go.Figure(data=data)
    fig.update_layout(height=650)
    fig.layout.paper_bgcolor = globalbgcolor
    fig.update_layout(margin=dict(l=20, r=20, t=35, b=25))

    return fig


@app.callback(Output('country_heading', 'children'),
              Output('area_info', 'children'),
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

    now_date = _datetime.datetime.today()
    perepis_date = _datetime.datetime(2021, 10, 1)
    otrezok = now_date - perepis_date
    otrezok.days
    date_df = tidy[tidy['Территория'].eq(area) & tidy['Год'].isin([2020, 2019, 2018])]
    date_df0 = date_df['Коэффициент общего прироста населения']
    pop_mean = date_df0.mean() / 1000
    init_pop = tidy[tidy['Территория'].eq(area) & tidy['Год'].eq(2021)]['Численность населения (по переписям), человек']
    init_pop = init_pop.reset_index().iloc[0, 1]
    now_pop = int(init_pop + (init_pop * pop_mean) * (otrezok.days / 365))
    now_pop = '{0:,}'.format(now_pop).replace(',', ' ')
    now_date_format = now_date.date().strftime("%d.%m.%Y")

    areainfo = mo_info[mo_info['Территория'].eq(area)]
    markdown = f"""
                 **{areainfo['Территория'].values[0]}** – {areainfo['Тип'].values[0]} {areainfo['Субъект РФ'].fillna('').values[0]}.
                  {areainfo['Частично'].fillna('').values[0]} ОКТМО: {areainfo['ОКТМО'].values[0]}.
                  Административный центр: {areainfo['Административный центр'].values[0]}. 
                  На 2021 г. население составило {areainfo['Население'].values[0]} чел. или {areainfo['Население процент'].values[0]}%
                  от всего населения Арктической зоны Российской Федерации.
                Оценка численности населения на сегодня ({now_date_format}): {now_pop} чел.
                """


    df = tidy[tidy['Территория'].isin(areas) & tidy['Год'].isin(years)]
    fig = px.line(df,
                  x='Год',
                  y=indicator,
                  title='<b>' + indicator + '</b><br>' + ', '.join(areas),
                  color='Территория', line_shape='spline',
                  markers=True)
    fig.update_layout(height=600)
    fig.layout.paper_bgcolor = globalbgcolor

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
    fig2.update_layout(margin=dict(l=20, r=20, t=25, b=25))

    edu = education
    edu_cols = edu.columns


    if areas[0] is None:
        raise PreventUpdate
    fig3 = px.bar(edu[edu['Территория']==areas[0]].dropna(),
                 x=edu_cols,
                 y='Год',
                 barmode='stack',
                 height=350,
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
    fig3.update_layout(margin=dict(l=20, r=20, t=30, b=25))

    table1 = profiles2[profiles2['Территория'] == areas[0]].iloc[:, 8:31]
    table2 = profiles2[profiles2['Территория'] == areas[0]].iloc[:, 31:54]
    table3 = profiles2.iloc[0:1, 8:31]
    table2.columns = table1.columns
    table = pd.concat([table1, table2, table3]).T.reset_index()


    if table.shape[1] == 4:
        table.columns = ['Показатель' , 'Значение в ' + areas[0], 'Ранг в АЗРФ', 'В целом по АЗРФ']
        table = dbc.Table.from_dataframe(table)
    else:
        table = html.Div()
    return 'Профиль: ' + area, markdown, fig, table0, fig2, fig3, table

app.title = "Цифровой двойник населения Арктики. Дашборд"
if __name__ == '__main__':
    app.run_server(debug=False)
