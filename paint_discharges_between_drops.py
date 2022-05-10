import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import os
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from tqdm.auto import tqdm
import re
from dash import Dash, callback_context
from dash import dcc, html
from dash.dependencies import Input, Output


sum_df = pd.read_csv('C:/Users/1/Desktop/VKR/Master/data/silicone SPBU/PKD_02.02.22_part/2/test_frames_drops'
                     '/cap_cut_crop_drops.mp4/sum_df_fix.cvs', index_col=0)
img = Image.open('C:/Users/1/Desktop/VKR/Master/data/silicone SPBU/PKD_02.02.22_part/2/test_frames_drops'
                 '/cap_cut_crop_drops.mp4/0000000000.jpg')
w, h = img.size



def au_frame(number):

    frame_name = '{0:0>10}.jpg'.format(number)
    path_frames = 'C:/Users/1/Desktop/VKR/Master/data/silicone SPBU/PKD_02.02.22_part/2/' \
                  'test_frames_drops/cap_cut_crop_drops.mp4/'
    frame_with_name = Image.open(f'{path_frames}{frame_name}')
    frame_with_name = frame_with_name.filter(ImageFilter.DETAIL)
    # enhancer = ImageEnhance.Brightness(frame_with_name)
    # frame_with_name = enhancer.enhance(1.2)

    return frame_with_name

#======================================================================================================================
# Построение карты разрядов


app = Dash(__name__)

t_range = np.sort(sum_df['t_sum'].unique())

app.layout = html.Div([
    html.H1('Разряды между каплями'),

    html.Div([
        dcc.Graph(id='plot_1')
    ], style={'float': 'center', 'width': '100%', 'display': 'inline-block',
              }),

    # html.Div([dcc.Input(id='find_frame', type='text', value=int(t_range.max()))],
    #           style={"display": "grid", "grid-template-columns": "10% 40% 10%"}),

    html.Div([
        html.Label(id='t-frame_name'),
        dcc.Slider(
            min=int(sum_df['t_sum'].min()),
            max=int(sum_df['t_sum'].max()),
            step=None,
            marks={int(t): {'label': ''} for i, t in enumerate(t_range)},
            value=int(40667),
            id='frame--slider',
            updatemode='mouseup',
        )], style={'width': '70%', 'display': 'inline-block', 'padding': '40px 20px 20px 20px'}),
    html.Button('<<', id='down', n_clicks=0),
    html.Button('>>', id='up', n_clicks=0),
])

n_frame = int(t_range.max())


@app.callback(
    [Output('plot_1', 'figure'),
     Output('t-frame_name', 'children'),],
    [Input('frame--slider', 'value'),
     Input('up', 'n_clicks'),
     Input('down', 'n_clicks')],
    prevent_initial_call=True)
def contr_plot(t_frame_slider, up, down):
    global sum_df, n_frame, t_range

    ctx = callback_context
    dropdown_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if dropdown_id == 'frame--slider':
        t_frame_dash = t_frame_slider
    elif dropdown_id == 'up':
        idx = np.where(t_range == n_frame)[0][0]
        t_frame_dash = t_range[idx + 1]
    elif dropdown_id == 'down':
        idx = np.where(t_range == n_frame)[0][0]
        t_frame_dash = t_range[idx - 1]

    n_frame = t_frame_dash

    # img_width = 229
    # img_height = 461

    # img_width = 268
    # img_height = 550

    # img_width = 259
    # img_height = 516

    # img_width = 248
    # img_height = 570

    plot_df = sum_df[sum_df.t_sum == t_frame_dash]
    print(plot_df.head(2))

    layout = go.Layout(autosize=False,
                       margin=dict(l=20, r=20, t=20, b=20),
                       width=2*w/4, height=h,
                       dragmode='drawrect')
    fig = go.Figure(layout=layout)

    fig.add_layout_image(
        dict(
            source=au_frame(t_frame_dash),
            xref="x",
            yref="y",
            x=0,
            y=0,
            sizex=w,
            sizey=h,
            sizing="stretch",
            opacity=1,
            layer="below"
        )
    )

    fig.add_trace(
        go.Scatter(x=plot_df['x_sum'],
                   y=-plot_df['y_sum'],
                   mode='markers',
                   marker=dict(color='red', size=1)
                   )
    )

    fig.update_xaxes(showgrid=False, range=(w/4, 3*w/4), )
    # fig.update_yaxes(autorange="reversed", showgrid=True, range=(0, img_height))
    fig['layout']['yaxis'].update(autorange=False, showgrid=True, range=[-h, 0])
    return [fig,
            f'Разряды между каплями на {t_frame_dash} фрейме']


if __name__ == '__main__':
    app.run_server(debug=True)


 # layout = go.Layout(margin=dict(l=20, r=20, t=20, b=20),
 #                       width=img_width, height=img_height)
 #    fig = go.Figure(data=[go.Scatter(x=plot_df['x_sum'],
 #                                     y=plot_df['y_sum'],
 #                                     mode='markers',
 #                                     marker=dict(color='red')
 #                                     )],
 #                    layout=layout)
 #
 #    fig.update_yaxes(autorange="reversed", showgrid=True,
 #                     tickvals=[0, img_height])