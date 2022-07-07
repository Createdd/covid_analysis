import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


final_df = pd.read_excel('data/cleaned_data.xlsx', index_col=0)
only_covid = final_df[final_df.covid_deaths.notna()]





# fig = px.line(
#     x=final_df.index,
#     y=final_df.nr_deaths,
#     title='Deaths per year'
# )

# fig.add_traces(
#     go.Scatter(
#         x=final_df.index, y=final_df.covid_deaths,
#         name='Covid deaths'))

# fig.show()

def draw_slope_line_at_point(fig, ind, x, y, slope_at_point, verbose=False):
    """Plot a line from an index at a specific point for x values, y values and their slopes"""


    y_low = (x[0] - x[ind]) * slope_at_point[ind] + y[ind]
    y_high = (x[-1] - x[ind]) * slope_at_point[ind] + y[ind]

    x_vals = [x[0], x[-1]]
    y_vals = [y_low, y_high]

    if verbose:
        print((x[0] - x[ind]))
        print(x[ind], x_vals, y_vals, y[ind],slope_at_point[ind])

    fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                name="Tangent at point",
                line = dict(color='orange', width=2, dash='dash'),
            )
        )

    return x_vals, y_vals



# ---

poly_degree = 10


y = only_covid.covid_deaths.values
x = np.arange(0, len(y))
x = x.reshape(-1, 1)

# just for checking with sklearn implementation
poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
poly_features = poly.fit_transform(x)
poly_reg_model = LinearRegression()
poly_reg_model.fit(poly_features, y)


#-----
fitted_params = np.polyfit(np.arange(0, len(y)), y, poly_degree )

polynomials = np.poly1d(fitted_params)

derivatives = np.polyder(polynomials)

y_value_at_point = polynomials(x).flatten()

slope_at_point = np.polyval(derivatives, np.arange(0, len(y)))
#-----



# x_range_ordinal_poly = np.linspace(x.min(), x.max(), len(y))

# poly_pred_x = poly.fit_transform(x_range_ordinal_poly.reshape(-1, 1))


# print(f'''
# x: {len(x), x},
# y: {len(y), y},
# ''')

# print(f'''
# fittedparams: {fitted_params, fitted_params},

# derivs: {derivatives},

# y vals at point: {y_value_at_point, len(y_value_at_point)},

# slope at point: {slope_at_point}''')

# ---

fig = px.scatter(
    x=only_covid.index,
    y=only_covid.covid_deaths,
    trendline="lowess",
    trendline_color_override="red",
    trendline_options=dict(frac=0.1),
    opacity=.5,
    title='Deaths per year'
)

traces = []
animation_dicts = dict()


traces.append(
            go.Scatter(
                x=np.arange(0, len(y)),
                y=only_covid.covid_deaths,
                name='Covid deaths',
                mode='lines',
                opacity=.5,
                line=dict(width=1.5)

            ))

# traces.append(
#             go.Scatter(
#                 x=np.arange(0, len(y)),
#                 y=only_covid.covid_deaths,
#                 mode='lines',
#                 opacity=.5,
#                 line={'shape': 'spline', 'smoothing': 1.3}
#             ))

traces.append(
            go.Scatter(
                x=np.arange(0, len(y)),
                y=only_covid.covid_deaths,
                name='Covid deaths',
                mode='markers',
                opacity=.5,
                line=dict(width=1)

            ))

traces.append(
    go.Scatter(
        x=np.arange(0, len(y)),
        #x=final_df.index.strftime('%Y-%m-%d').to_list(),
        y=y_value_at_point,
        name='Polynomial regression Fit',
                    mode='lines',
                opacity=.5,
                line=dict(width=1.5)
    ))



for pt in np.arange(0, len(y)):#[31, 60]:
    x_vals, y_vals = draw_slope_line_at_point(
        fig,
        x= np.arange(0, len(y)),
        y = y_value_at_point,
        slope_at_point=slope_at_point,
        ind = pt)

    animation_dicts[pt]= [x_vals, y_vals]

# plotting tangent lines without animation
#     traces.append(
#             go.Scatter(
#                 x=x_vals,
#                 y=y_vals,
#                 mode='lines',
#                 opacity=.4,
#                 line=dict(width=1.5)))

#     fig.add_annotation(
#         x=pt,
#         y=y_value_at_point[pt],
#         text=f'''Slope: {slope_at_point[pt]:.2f}\t {only_covid.index.strftime('%Y-%m-%d')[pt]}''',
#         showarrow=True,
#         arrowhead=1)


frame_data = []
slider_steps = []

for k in range(0, len(final_df)):

# plotting the scatterplot as well in frames
#     frame_data.append(
#         dict(data=

#             [dict(
#                 type = 'scatter',
#                 x=np.arange(0, len(y))[:k],
#                 y=only_covid.covid_deaths[:k]
#                 )]
#             )
#     )


    # add slope lines
    if k in animation_dicts.keys():
        frame_data.append(
            dict(data=
                [dict(
                    type = 'scatter',
                    x=animation_dicts[k][0],
                    y=animation_dicts[k][1],
                    mode='lines',
                    line={'dash': 'dash', 'color': 'green'}
                )]
            )
        )

        slider_steps.append(
            {"args": [
                frame_data[k],
                {"frame": {"duration": 200, "redraw": False},
                 "mode": "immediate",
                 "transition": {"duration": 200}}
                ],
            "label": only_covid.index.strftime('%Y-%m-%d')[k],
            "method": "animate"
            }
       )



all_frames = frame_data
frames=all_frames

sliders_dict = {
    "active": 1,
    "yanchor": "top",
    "xanchor": "left",

    "currentvalue": {
        "font": {"size": 20},
        "prefix": "Date:",
        "visible": True,
        "xanchor": "right"
    },
    "transition": {"duration": 200, "easing": "cubic-in-out"},
    "pad": {"b": 10, "t": 50},
    "len": 0.9,
    "x": 0.1,
    "y": 0,
    "steps": slider_steps
}


layout = go.Layout(
        xaxis={"range": [0, len(y)], "title": "weeks"},
                    sliders=[sliders_dict],
#                    showlegend=False,
#                    hovermode='x unified',
                   updatemenus=[
                        dict(
                            type='buttons',
                            buttons=[
                                dict(
                                    label='Show tangents',
                                    method='animate',
                                    args=[None,
                                      dict(frame=dict(duration=200,
                                                      redraw=False),
                                                      transition=dict(duration=0),
                                                      fromcurrent=True,
                                                      mode='immediate')
                                 ]),
                                {
                                    "args": [[None], {"frame": {"duration": 200, "redraw": False},
                                                      "mode": "immediate",
                                                      "transition": {"duration": 200}}],
                                    "label": "Pause",
                                    "method": "animate"
                                }
                            ],

                            direction="left",
                            pad= {"r": 10, "t": 87},
                            showactive= False,
                            x=0.1,
                            xanchor= "right",
                            y= 0,
                            yanchor= "top"
                        )
                    ]
                  )



fig = go.Figure(
    data=traces,

    layout=layout,
    frames=frames)

# fig.show()


sidebar = st.sidebar
show_data = sidebar.checkbox("Show Data")

if show_data:
    st.dataframe(only_covid)

st.plotly_chart(fig)