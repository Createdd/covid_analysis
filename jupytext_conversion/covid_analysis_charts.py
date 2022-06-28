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

# +
import numpy as np
import pandas as pd

from pandas_profiling import ProfileReport
import plotly.express as px
import plotly.graph_objects as go

# -

# ## Load data
#
# general deaths: https://www.data.gv.at/katalog/dataset/stat_gestorbene-in-osterreich-ohne-auslandssterbefalle-ab-2000-nach-kalenderwoche/resource/b0c11b84-69d4-4d3a-9617-5307cc4edb73
#
# covid deaths: https://www.data.gv.at/katalog/dataset/covid-19-zeitverlauf-der-gemeldeten-covid-19-zahlen-der-bundeslander-morgenmeldung/resource/24f56d99-e5cc-42e4-91fa-3e6a06b73064?view_id=89e19615-c7b3-445c-9924-e9dd6b8ace75

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

profile = ProfileReport(df, title="Pandas Profiling Report")
profile2 = ProfileReport(df_deaths, title="Pandas Profiling Report")

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
# -

import datetime
df['conv_date']= df.date.map(datetime.datetime.toordinal)


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

grpd_date_df_deaths

fig = px.line(
    x=grpd_date_df_deaths.index,
    y=grpd_date_df_deaths.deaths_per_day, 
    title='Deaths per year'
)
fig.show()

# +
# grpd_date_df_deaths['date'] = grpd_date_df_deaths.index
# grpd_date_df_deaths
# -

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

# +
trace1 = go.Scatter(
                    x=final_df.index,
                    y=final_df.nr_deaths,
                    mode='lines',
                    line=dict(width=1.5))


frames=[
    dict(
        data=[
            dict(
                type = 'scatter',
                x=final_df.index[:k],
                y=final_df.nr_deaths[:k])]
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
fig = go.Figure(data=[trace1], frames=frames, layout=layout)
# a
fig.show()
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

# regression params not available for lowess
# -

# ### Polynomial Regression

# +

fig = px.scatter(
    x=final_df.index,
    y=final_df.nr_deaths, 
    trendline="lowess", #ols
    trendline_color_override="red",
    trendline_options=dict(frac=0.1),
    opacity=.5,
    title='Deaths per year'
)
fig.show()

# regression params not available for lowess
# -

# ### Sklearn Implementation

# +
from sklearn.linear_model import LinearRegression


x=final_df.index.values


y=final_df.nr_deaths.values
x = np.arange(0, len(y))

# x = final_df.conv_date.values
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
        x=final_df.index, y=y_range, 
        name='Regression Fit'))


fig.show()


# +
from sklearn.preprocessing import PolynomialFeatures

poly_degree = 3


y = final_df.nr_deaths.values

x = np.arange(0, len(y))
x = x.reshape(-1, 1)

poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
poly_features = poly.fit_transform(x)

poly_reg_model = LinearRegression()
poly_reg_model.fit(poly_features, y)

#-----
fittedParameters = np.polyfit(np.arange(0, len(y)), y, poly_degree )

poly_new = np.poly1d(fittedParameters)

deriv = np.polyder(poly_new)

# y_value_at_point = np.polyval(fittedParameters, y)
y_value_at_point = poly_new(x).flatten()


slope_at_point = np.polyval(deriv, np.arange(0, len(y)))
#-----



x_range_ordinal_poly = np.linspace(x.min(), x.max(), len(y))

poly_pred_x = poly.fit_transform(x_range_ordinal_poly.reshape(-1, 1))


y_range_poly = poly_reg_model.predict(
    poly_pred_x)


print(y_range_poly.shape, poly_pred_x.shape)



print(poly_reg_model.intercept_)
print(poly_reg_model.coef_)
print(f'''
x: {len(x), x},
y: {len(y), y},
''')
print(f'''
fittedparams: {fittedParameters, fittedParameters},

derivs: {deriv},

y vals at point: {y_value_at_point, len(y_value_at_point)},

slope at point: {slope_at_point}''')


# +
def slope_line(fig, ind, x, y):
    """Plot a line from slope and intercept"""
    
    
#     y_value_at_point = np.polyval(fittedParameters, y[ind])
#     slope_at_point = np.polyval(deriv, y[ind])

    
    ylow = (x[0] - x[ind]) * slope_at_point[ind] + y[ind]
    yhigh = (x[-1] - x[ind]) * slope_at_point[ind] + y[ind]
    
    x_vals = [x[0], x[-1]]
#     y_vals = [y[ind] , x[ind +1] * slope_at_point[ind] + intercept[ind]]
    y_vals = [ylow, yhigh]
    
    

    print(x[ind], x_vals, y_vals, y[ind],slope_at_point[ind])
    
    fig.add_trace(
            go.Scatter(
                x=x_vals, 
                y=y_vals, 
                name="Tangent at point", 
                line = dict(color='orange', width=2, dash='dash')
            )
        )
# -

y_value_at_point.flatten()#.shape, y_range_poly.shape

np.arange(0, len(y))

# +
fig = px.scatter(
    x=np.arange(0, len(y)),
    y=final_df.nr_deaths, 
    opacity=.5,
#     trendline='ols', trendline_color_override='darkblue',
    title='Deaths per year'
)


# x_range = pd.date_range(start=final_df.index[0],
#                   end=final_df.index[-1],
#                   periods=len(final_df.index.values))
fig.add_traces(
    go.Scatter(
        x=np.arange(0, len(y)), y=y_range_poly, 
        name='Polynomial regression Fit'))

fig.add_traces(
    go.Scatter(
        x=np.arange(0, len(y)), y=y_value_at_point, 
        name='Polynomial regression Fit 2'))




for pt in [1000, 10]:
    slope_line(
        fig, 
        x= np.arange(0, len(y)), 
        y = y_value_at_point, 
        ind = pt)

# fig.update_layout(yaxis_range=[0,2000])

fig.show()

# -




