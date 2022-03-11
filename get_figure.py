import pandas as pd
import plotly.graph_objects as go
import numpy as np
from tqdm.auto import tqdm
from dash import Dash
from dash import dcc, html
from dash.dependencies import Input, Output
from PIL import Image

sum_df = pd.read_csv('C:/Users/1/Desktop/ВКР/Магистерская/data/silicone SPBU/PKD_18.02.22_part/10'
                     '/test_frames/cap_cut_crop.mp4/sum_df.cvs', index_col=0)
sum_df = sum_df.sort_values('t_sum')

path_image0_p = 'C:/Users/1/Desktop/ВКР/Магистерская/data/silicone SPBU/PKD_18.02.22_part/10/test_frames' \
                '/cap_cut_crop.mp4/0000000000.jpg'

img_p = Image.open(path_image0_p)
w, h = img_p.size

def f(x_m, y_m, plot_df):
    alpha_sum_f = np.zeros((y_m.shape[0], x_m.shape[0]))
    for y, x, alpha in tqdm(plot_df.values):
        alpha_sum_f[y, x] = alpha
    return alpha_sum_f

def log10f(x_m, y_m, plot_df):
    alpha_sum_f = np.zeros((y_m.shape[0], x_m.shape[0]))
    for y, x, alpha in tqdm(plot_df.values):
        alpha_sum_f[y, x] = np.log10(alpha)
    alpha_sum_f[alpha_sum_f <= 1] = 1
    return alpha_sum_f


# Построение карты разрядов
app = Dash(__name__)

t_range = sum_df['t_sum'].unique()
t_range = np.arange(t_range.min(), t_range.max(), 100)

app.layout = html.Div([
    html.H1('Карта воздействия разрядов на поверхности образца'),

    html.Div([
        dcc.Graph(id='plot_1')
    ], style={'float': 'center', 'width': '100%', 'display': 'inline-block',
              }),

    html.Div([
        html.Label(id='t-frame_name'),
        dcc.Slider(
            min=int(sum_df['t_sum'].min()),
            max=int(sum_df['t_sum'].max()),
            step=None,
            marks={int(t): {'label': ''} for i, t in enumerate(t_range)},
            value=62500,
            id='frame--slider',
            updatemode='mouseup'
        )], style={'width': '50%', 'display': 'inline-block', 'padding': '40px 20px 20px 20px'}),
])

# app.layout = html.Div([
#     html.H1('Карта воздействия разрядов на поверхности образца'),

#     html.Div([
#         html.Div([
#             html.H3('Column 1'),
#             dcc.Graph(id='plot_1', figure=fig)
#         ], className="six columns"),
#
#         html.Div([
#             html.H3('Column 2'),
#             dcc.Graph(id='plot_2', figure={'data': [{'y': [1, 2, 3]}]})
#         ], className="six columns"),
#     ], className="row"),
#
#     html.Div([
#         html.Label(id='t-frame_name'),
#         dcc.Slider(
#             min=int(sum_df['t_sum'].min()),
#             max=int(sum_df['t_sum'].max()),
#             step=None,
#             marks={int(t): {'label': ''} for i, t in enumerate(t_range)},
#             value=62500,
#             id='frame--slider',
#             updatemode='mouseup'
#         )], style={'width': '50%', 'display': 'inline-block', 'padding': '40px 20px 20px 20px'}),
# ])


@app.callback(
    [Output('plot_1', 'figure'),
     Output('t-frame_name', 'children'),],
    [Input('frame--slider', 'value'),])
def contr_plot(t_frame):
    global sum_df

    img_width = 229
    img_height = 461

    plot_df = sum_df[sum_df.t_sum <= t_frame].groupby(
        by=['y_sum', 'x_sum'], as_index=False)['alpha_sum'].sum()

    dx = 1
    dy = 1
    # w = 203
    # h = 444

    # w = 253
    # h = 511

    # w = 249
    # h = 527



    # w = 237
    # h = 527

    x_m = np.arange(0, w, dx)
    y_m = np.arange(0, h, dy)

    fig = go.Figure(
        go.Contour(
            z=f(x_m, y_m, plot_df).tolist(),
            dx=dx,
            x0=0,
            dy=dy,
            y0=0,
            colorscale='Jet',#"Cividis",
            line=dict(width=0),
            ncontours=100,
           #  colorbar=dict(
           #      title='Log_10'
           # )
            #contours=dict(start=10)
        )
    )

    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
                      margin=dict(l=20, r=20, t=20, b=20),
                      width=w*1.5, height=h*1.5,
    )

    return [fig,
            f'Суммирование яркости до {t_frame} фрейма']


if __name__ == '__main__':
    app.run_server()
