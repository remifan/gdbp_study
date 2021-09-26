from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def py_params(p, title, margin):
  p = p.astype({'fs': 'float',
    'D0': 'complex64',
    'D1': 'complex64',
    'D2': 'complex64',
    'N0': 'complex64',
    'N1': 'complex64',
    'N2': 'complex64',
    'R':  'complex64',})
   
  fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                      specs=[[{"secondary_y": True}],
                             [{"secondary_y": True}],
                             [{"secondary_y": True}]])

  for i in range(3):
    fig.add_trace(go.Scatter(
        x=p.fs,
        y=np.abs(p[f'D{i}']),
        mode='lines',
        name=f'|D{i+1}|',
    ), row=1, col=1, secondary_y=False)

    fig.add_trace(go.Scatter(
        x=p.fs,
        y=np.unwrap(np.angle(p[f'D{i}'])),
        mode='lines',
        line={'dash': 'dash'},
        name=f'∠D{i+1}',
    ), row=1, col=1, secondary_y=True)


  for i in range(3):
    fig.add_trace(go.Scatter(
        x=p.fs,
        y=np.abs(p[f'N{i}']),
        mode='lines',
        name=f'|N{i+1}|',
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=p.fs,
        y=np.unwrap(np.angle(p[f'N{i}'])),
        mode='lines',
        line={'dash': 'dash'},
        name=f'∠N{i+1}',
    ), row=2, col=1, secondary_y=True)


  fig.add_trace(go.Scatter(
      x=p.fs,
      y=np.abs(p[f'R']),
      mode='lines',
      name=f'|R|',
  ), row=3, col=1)

  fig.add_trace(go.Scatter(
      x=p.fs,
      y=np.unwrap(np.angle(p[f'R'])),
      mode='lines',
      line={'dash': 'dash'},
      name=f'∠R',
  ), row=3, col=1, secondary_y=True)

  fig.update_layout(height=800,
                    title_text=title,
                    margin=margin,
                    template='plotly_white')
  # fig.update_xaxes(title_text="freq. (GHz)")
  fig.layout.xaxis3.update(title="freq. (GHz)")
  # single axis-label trick, https://plot.ly/~empet/14983
  fig.layout.yaxis3.update(title='amplitude (a.u.)')
  fig.layout.yaxis4.update(title='phase (rad)')
  fig.show()


def py_params_few_taps(p, title, margin):
  p = p.astype({'fs': 'float',
    'D0': 'complex64',
    'D1': 'complex64',
    'D2': 'complex64',
    'N0': 'complex64',
    'N1': 'complex64',
    'N2': 'complex64',
    'R':  'complex64',})
   
  fig = make_subplots(rows=1, cols=2, shared_xaxes=False,
                      specs=[[{"secondary_y": True}, {"secondary_y": True}]])

  for i in range(3):
    fig.add_trace(go.Scatter(
        x=p.fs,
        y=np.abs(p[f'D{i}']),
        mode='lines',
        name=f'|D{i+1}|',
    ), row=1, col=1, secondary_y=False)

    fig.add_trace(go.Scatter(
        x=p.fs,
        y=np.unwrap(np.angle(p[f'D{i}'])),
        mode='lines',
        line={'dash': 'dash'},
        name=f'∠D{i+1}',
    ), row=1, col=1, secondary_y=True)


  for i in range(3):
    fig.add_trace(go.Scatter(
        x=p.fs,
        y=np.abs(p[f'N{i}']),
        mode='lines',
        name=f'|N{i+1}|',
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=p.fs,
        y=np.unwrap(np.angle(p[f'N{i}'])),
        mode='lines',
        line={'dash': 'dash'},
        name=f'∠N{i+1}',
    ), row=1, col=2, secondary_y=True)

  fig.update_layout(height=250,
                    title_text=title,
                    margin=margin,
                    template='plotly_white')
  fig.update_xaxes(title_text="freq. (GHz)")
  # single axis-label trick, https://plot.ly/~empet/14983
  fig.layout.yaxis1.update(title='amplitude (a.u.)')
  fig.layout.yaxis4.update(title='phase (rad)')
  return fig


def py_qvstaps(df_test_res):
    blues = [[0, "rgb(107,184,255)"], 
            [1, "rgb(0,90,124)"]]

    reds = [[0, "rgb(255,107,184)"], 
            [1, "rgb(128,0,64)"]]

    yellows = [[0, "rgb(245,250,42)"], 
            [1, "rgb(250,171,42)"]]

    colors = [yellows, blues, reds]

    camera = dict(
        eye=dict(x=1.75, y=-1.1, z=1.2)
    )

    dat = []

    for mod, c in zip(['FDBP', 'EDBP', 'GDBP'], colors):
        df = df_test_res.groupby('Model').get_group(mod).pivot('dtaps', 'ntaps', 'Q')
        (X, Y), Z = np.meshgrid(df.columns.values, df.index.values), df.values
        dat.append(go.Surface(x=X, y=Y, z=Z,
                              name=mod,
                              colorscale=c,
                              opacity=0.8,
                              showscale=False,
                              showlegend=True,
                              hidesurface=False,
                              contours_x=dict(show=True),
                              contours_y=dict(show=True)))

    fig = go.Figure(data=dat)

    fig.update_layout(autosize=False,
                      width=700, height=600,
                      margin=dict(l=65, r=50, b=0, t=0),
                      legend=dict(
                          yanchor="top",
                          y=0.93,
                          xanchor="right",
                          x=1.1
                      ),
                      scene = dict(
                          camera=camera,
                          xaxis_title='N-filter length',
                          yaxis_title='D-filter length',
                          zaxis_title='Q-factor (dB)'))

    fig.show()

