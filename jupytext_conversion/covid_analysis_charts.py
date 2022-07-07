# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python [conda env:streamlit_apps] *
#     language: python
#     name: conda-env-streamlit_apps-py
# ---

# # Analysing Austrias deaths and their Covid-19 correlation
#
#
# The data is used from **data.gv.at** which is Austrias official open data source. 
#
# More about it [here](https://www.data.gv.at/infos/zielsetzung-data-gv-at/)
#
# ## Datasets
#
# The used datasets are from:
#
# - general deaths: https://www.data.gv.at/katalog/dataset/stat_gestorbene-in-osterreich-ohne-auslandssterbefalle-ab-2000-nach-kalenderwoche/resource/b0c11b84-69d4-4d3a-9617-5307cc4edb73
#
# - covid deaths: https://www.data.gv.at/katalog/dataset/covid-19-zeitverlauf-der-gemeldeten-covid-19-zahlen-der-bundeslander-morgenmeldung/resource/24f56d99-e5cc-42e4-91fa-3e6a06b73064?view_id=89e19615-c7b3-445c-9924-e9dd6b8ace75
#
# ## Goal of the analysis
#
# This project has multiple goals.
#
# - Work with Austriaas open source data and see its ease of use
# - Plot interactively the development of deaths and covid deaths over time
# - Fit a (polynomial) regression through the data and calculate the slopes and given points to see the change of deaths rates
#
# # Analysis

# +
import datetime
import numpy as np
import pandas as pd

from pandas_profiling import ProfileReport

import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# +
import plotly.io as pio

# for rendering plot static
# pio.renderers.default = "svg"
# -

# ## Load data
#
#

# +
data_file_name = 'data/OGD_gest_kalwo_GEST_KALWOCHE_100.csv'
covid_deaths_data_file_name = 'data/timeline-faelle-bundeslaender.csv'

df = pd.read_csv(data_file_name, sep=';')  
df_deaths = pd.read_csv(covid_deaths_data_file_name, sep=';')  
# -

df

df_deaths

# ## Pandas Profiling
#

# +
# profile = ProfileReport(df, title="Pandas Profiling Report")
# profile2 = ProfileReport(df_deaths, title="Pandas Profiling Report")

# +
# profile2
# -

# ## Data Cleaning
#

# ### Clean Dates

df = df.rename(columns={"C-KALWOCHE-0": "cal_week", "C-B00-0": "state", "C-C11-0": "gender", "F-ANZ-1": "counts" })

# +
df['year'] = [d.split("-")[1][:4] for d in df['cal_week']]
df['cal_week'] = [d.split("-")[1][4:] for d in df['cal_week']]
df['formatted_date'] = df.year.astype(str)  + df.cal_week.astype(str) + '0'
df['date'] = pd.to_datetime(df['formatted_date'], format='%Y%W%w')

# df['datetime'] = pd.to_datetime(df.year.astype(str) + '-' + 
#                                 df.cal_week.astype(str) + '-1' , format="%Y-%W-%w").dt.strftime('%Y-%W')

# +

df['conv_date']= df.date.map(datetime.datetime.toordinal)

# -

df

# ### Aggregate dates

df.groupby('date').agg('sum')

grpd_date = df.groupby('date').agg('sum') 
grpd_date = grpd_date.rename(columns={'counts':'nr_deaths'}) 


# +
# grpd_date['date'] = grpd_date.index
# grpd_date
# -

# ### Clean 2nd dataset

temp = df_deaths.copy(deep=True)

df_deaths = temp.copy(deep=True)

df_deaths = df_deaths[df_deaths.Name == 'Österreich']

df_deaths

df_deaths = df_deaths.rename(columns={'Datum':'formatted_date', 'Todesfaelle':'deaths'})


df_deaths = df_deaths[['formatted_date', 'deaths']]
df_deaths

# +
df_deaths['formatted_date'] = [d.split("T")[0] for d in df_deaths['formatted_date']]

df_deaths['date'] = pd.to_datetime(df_deaths['formatted_date'])

# df_deaths['formatted_date'] = pd.to_datetime(df_deaths['formatted_date'])
# df_deaths['formatted_date'] = df_deaths['formatted_date'].dt.tz_localize('CET', utc=True)
 
    
# df_deaths['date'] =  df_deaths['formatted_date'].dt.strftime('%Y-%m-%d')
# -

df_deaths

grpd_date_df_deaths = df_deaths.groupby('date').agg('sum') 

grpd_date_df_deaths['deaths_per_day'] = grpd_date_df_deaths.diff()

# +
# https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(21)02796-3/fulltext#seccestitle150
# austria has ratio of 1.33
# grpd_date_df_deaths['deaths_per_day_corrected'] = grpd_date_df_deaths.deaths_per_day * 1.33 
# -

grpd_date_df_deaths

fig = px.line(
    x=grpd_date_df_deaths.index,
    y=grpd_date_df_deaths.deaths_per_day, 
    title='Deaths per year'
)
fig.show()

# ### Merge datasets

total_df = pd.merge(grpd_date, grpd_date_df_deaths, how='left', on='date')
total_df

# +
total_df = total_df.rename(
    columns={'deaths_per_day':'covid_deaths'})

total_df = total_df[['nr_deaths', 'covid_deaths']]
# -

total_df

# ### Remove outliers

total_df.describe()

# +
upper_limit = total_df['nr_deaths'].quantile(0.99999)
lower_limit = total_df['nr_deaths'].quantile(0.00001)

new_df = total_df[(
    total_df['nr_deaths'] <= upper_limit) & (
    total_df['nr_deaths'] >= lower_limit)]
# -

new_df

final_df = new_df

only_covid = final_df[final_df.covid_deaths.notna()]
only_covid.shape

# +
# only if we want to limit all deaths to covid time period
# final_df = only_covid

# +
# final_df.to_excel("data/cleaned_data.xlsx")
# -

# ## Data visualization - All deaths
#

# +
fig = px.line(
    x=final_df.index,
    y=final_df.nr_deaths, 
    title='Deaths per year'
)

fig.add_traces(
    go.Scatter(
        x=final_df.index, y=final_df.covid_deaths, 
        name='Covid deaths'))

fig.show()

# + code_folding=[]
trace1 = go.Scatter(
                    x=final_df.index,
                    y=final_df.nr_deaths,
                    mode='lines',
                    line=dict(width=1.5))

trace2 = go.Scatter(
        x=final_df.index, y=final_df.covid_deaths, 
        name='Covid deaths')


frames=[
    dict(
        data=[
            dict(
                type = 'scatter',
                x=final_df.index[:k],
                y=final_df.nr_deaths[:k]),
    
            dict(
                type = 'scatter',
                x=final_df.index[:k],
                y=final_df.covid_deaths[:k])]
    )
    for k in range(0, len(final_df))]

layout = go.Layout(width=1000,
                   height=600,
                   showlegend=False,
                   hovermode='x unified',
                   updatemenus=[
                        dict(
                            type='buttons', showactive=False,
                            y=1.05,
                            x=1.15,
                            xanchor='right',
                            yanchor='top',
                            pad=dict(t=0, r=10),
                            buttons=[dict(label='Build line',
                            method='animate',
                            args=[None, 
                                  dict(frame=dict(duration=2, 
                                                  redraw=False),
                                                  transition=dict(duration=0),
                                                  fromcurrent=True,
                                                  mode='immediate')]
                            )]
                        ),
                        dict(
                            type = "buttons",
                            direction = "left",
                            buttons=list([
                                dict(
                                    args=[{"yaxis.type": "linear"}],
                                    label="LINEAR",
                                    method="relayout"
                                ),
                                dict(
                                    args=[{"yaxis.type": "log"}],
                                    label="LOG",
                                    method="relayout"
                                )
                            ]),
                        ),
                    ]              
                  )
# layout.update(xaxis =dict(range=['2020-03-16', '2020-06-13'], autorange=False),
#               yaxis =dict(range=[0, 35000], autorange=False));
fig = go.Figure(data=[trace1, trace2], frames=frames, layout=layout)

# fig.show()
# -

# ### Simple regression
#
# Ordinar Least Squares

# +

fig = px.scatter(
    x=final_df.index,
    y=final_df.nr_deaths, 
    trendline="ols",
    trendline_color_override="red",
    opacity=.5,
    title='Deaths per year'
)
fig.show()

results = px.get_trendline_results(fig)
results = results.iloc[0]["px_fit_results"].summary()
print(results)
# regression params not available for lowess
# -

# #### Sklearn Implementation

# +
x=final_df.index.values

y=final_df.nr_deaths.values
x = np.arange(0, len(y))
x = x.reshape(-1, 1)

model = LinearRegression()
model.fit(x, y)


x_range_ordinal = np.linspace(x.min(), x.max(), len(y))
y_range = model.predict(x_range_ordinal.reshape(-1, 1))

len(x_range_ordinal), len(y_range)

# +

fig = px.scatter(
    x=final_df.index,
    y=final_df.nr_deaths, 
    opacity=.5,
#     trendline='ols', trendline_color_override='darkblue',
    title='Deaths per year'
)


fig.add_traces(
    go.Scatter(
        x=final_df.index, 
        y=y_range, 
        name='Regression Fit'))


fig.show()

# -

# ### Polynomial Regression

# +

fig = px.scatter(
    x=final_df.index,
    y=final_df.nr_deaths, 
    trendline="lowess", 
    trendline_color_override="red",
    trendline_options=dict(frac=0.1),
    opacity=.5,
    title='Deaths per year'
)
fig.show()

# regression params not available for lowess

# +
poly_degree = 10

y = final_df.nr_deaths.values

x = np.arange(0, len(y))
x = x.reshape(-1, 1)

# just for checking with sklearn implementation
poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
poly_features = poly.fit_transform(x)
poly_reg_model = LinearRegression()
poly_reg_model.fit(poly_features, y)


# y_range_poly = poly_reg_model.predict(
#     poly_pred_x)
# print(poly_reg_model.intercept_)
# print(poly_reg_model.coef_)
###

#-----
fitted_params = np.polyfit(np.arange(0, len(y)), y, poly_degree )

polynomials = np.poly1d(fitted_params)

derivatives = np.polyder(polynomials)

y_value_at_point = polynomials(x).flatten()

slope_at_point = np.polyval(derivatives, np.arange(0, len(y)))
#-----



# x_range_ordinal_poly = np.linspace(x.min(), x.max(), len(y))

# poly_pred_x = poly.fit_transform(x_range_ordinal_poly.reshape(-1, 1))


print(f'''
x: {len(x), x},
y: {len(y), y},
''')

print(f'''
fittedparams: {fitted_params, fitted_params},

derivs: {derivatives},

y vals at point: {y_value_at_point, len(y_value_at_point)},

slope at point: {slope_at_point}''')


# -

# #### Calculate polynomial slope at certain point

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

# + code_folding=[]
fig = px.scatter(
    x=np.arange(0, len(y)),
    #x=final_df.index,
    y=final_df.nr_deaths, 
    opacity=.3,
    title='Deaths per year'
)


fig.add_traces(
    go.Scatter(
        x=np.arange(0, len(y)),
        #x=final_df.index.strftime('%Y-%m-%d').to_list(),
        y=y_value_at_point, 
        name='Polynomial regression Fit 2'))


# Replace x axis ticks with dates instead numbers
# fig.update_xaxes(
#     rangeslider_visible=True,
#     ticktext=final_df.index.strftime('%Y-%m-%d')[::10],
#     tickvals=np.arange(0, len(y))[::10],
    #tickformatstops not working with ticktextZ
# )


for pt in [750 ,1080]:
# for pt in [31]:
    draw_slope_line_at_point(
        fig, 
        x= np.arange(0, len(y)),
        #x=final_df.index,
        y = y_value_at_point,
        slope_at_point=slope_at_point,
        ind = pt)
    
    fig.add_annotation(x=pt, y=y_value_at_point[pt],
            text=f'''Slope: {slope_at_point[pt]:.2f}\t {final_df.index.strftime('%Y-%m-%d')[pt]}''',
            showarrow=True,
            arrowhead=1)


fig.update_layout(
    hovermode='x unified',
)
    
fig.show()
# pio.write_html(fig, file='all_deaths_with specific_slopes.html', auto_open=False)
# -

# ## Visualization - Covid deaths

fig = px.scatter(
    x=only_covid.index,
    y=only_covid.covid_deaths, 
    trendline="ols",
    trendline_color_override="red",
    opacity=.5,
    title='Deaths per year'
)
fig.show()

# +
fig = px.scatter(
    x=only_covid.index,
    y=only_covid.covid_deaths, 
    trendline="lowess",
    trendline_color_override="red",
    trendline_options=dict(frac=0.1),
    opacity=.5,
    title='Deaths per year'
)
fig.show()

# regression params not available for lowess
# -

# ### Polynomial
#

# +
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

print(f'''
fittedparams: {fitted_params, fitted_params},

derivs: {derivatives},

y vals at point: {y_value_at_point, len(y_value_at_point)},

slope at point: {slope_at_point}''')

# +
fig = px.scatter(
    x=np.arange(0, len(y)),
    y=only_covid.covid_deaths, 
    opacity=.5,
    title='Deaths per year'
)


fig.add_traces(
    go.Scatter(
        x=np.arange(0, len(y)),
        #x=final_df.index.strftime('%Y-%m-%d').to_list(),
        y=y_value_at_point, 
        name='Polynomial regression Fit 2'))

for pt in [31]:
    draw_slope_line_at_point(
        fig, 
        x= np.arange(0, len(y)),
        #x=final_df.index,
        y = y_value_at_point,
        slope_at_point=slope_at_point,
        ind = pt)
    
    fig.add_annotation(
        x=pt, 
        y=y_value_at_point[pt],
        text=f'''Slope: {slope_at_point[pt]:.2f}\t {only_covid.index.strftime('%Y-%m-%d')[pt]}''',
        showarrow=True,
        arrowhead=1)


fig.update_layout(
    hovermode='x unified',
)
    
fig.show()
# -

# ### Interactive tangent visualization

# + code_folding=[]
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

fig.show()
# -

only_covid

# +
# pio.write_html(fig, file='covid_deaths_with tangents.html', auto_open=False)
# -

# # Conclusion
#
#
# A proper interpretation would need a lot more research. As stated above this notebook as also technical implementational goals instead of merely analysing the deaths of people in Austria. 
#
# I, as an average educated person in this matter witness that
# - the dataset needs preparation to be used efficiently for analysis
# - analysis of the death rates alone is not sufficient to tell about a strong increase of deaths because of corona 
# - ( see a more detailed analysis here: https://www.data.gv.at/anwendungen/wiener-mortalitaetsmonitoring/ )
# - looking at the death rates alone it becomes clear that covid might very well contribute to the fact that there is an increase in deaths in the last 2 years
#


