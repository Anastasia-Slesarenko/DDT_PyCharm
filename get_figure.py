import pandas as pd
import plotly.graph_objects as go
import numpy as np
from tqdm.auto import tqdm
from dash import Dash
from dash import dcc, html
from dash.dependencies import Input, Output

sum_df = pd.read_csv('C:/Users/1/Desktop/ВКР/Магистерская/'
                     'ProjectPyCharm/DDT_PyCharm/test_frames/cap_cut_crop.mp4/sum_df.cvs', index_col=0)
sum_df = sum_df.sort_values('t_sum')


def f(x_m, y_m, plot_df):
    alpha_sum_f = np.zeros((y_m.shape[0], x_m.shape[0]))
    for y, x, alpha in tqdm(plot_df.values):
        alpha_sum_f[y, x] = alpha
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
            marks={int(t): {'label': f'{round(t / t_range.max())}%'} for t in t_range},
            value=62500,
            id='frame--slider',
            updatemode='mouseup'
        )], style={'width': '50%', 'display': 'inline-block', 'padding': '40px 20px 20px 20px'}),
])


@app.callback(
    [Output('plot_1', 'figure'),
     Output('t-frame_name', 'children'),],
    [Input('frame--slider', 'value'),])
def contr_plot(t_frame):
    global sum_df

    plot_df = sum_df[sum_df.t_sum <= t_frame].groupby(
        by=['y_sum', 'x_sum'], as_index=False)['alpha_sum'].sum()

    dx = 1
    dy = 1
    # w = 203
    # h = 444
    x_m = np.arange(0, 203, dx)
    y_m = np.arange(0, 444, dy)

    fig = go.Figure(
        go.Contour(
            z=f(x_m, y_m, plot_df).tolist(),
            dx=dx,
            x0=0,
            dy=dy,
            y0=0,
            colorscale="Cividis",
            line=dict(width=0),
            ncontours=100
        )
    )

    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
                      margin=dict(l=20, r=20, t=20, b=20),
                      width=96 * 250/60, height=96 * 400/60,
    )

    return [fig,
            f'Суммирование яркость до {t_frame} фрейма']


if __name__ == '__main__':
    app.run_server()
