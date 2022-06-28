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


# +
data_file_name = 'data/OGD_gest_kalwo_GEST_KALWOCHE_100.csv'

df = pd.read_csv(data_file_name, sep=';')  
# -

df

# ## Pandas Profiling
#

profile = ProfileReport(df, title="Pandas Profiling Report")


profile

# ## Data Cleaning
#

df = df.rename(columns={"C-KALWOCHE-0": "cal_week", "C-B00-0": "state", "C-C11-0": "gender", "F-ANZ-1": "counts" })

# +
# df['year'] = [d.split("-")[1][:4] for d in df['cal_week']]
# df['cal_week'] = [d.split("-")[1][4:] for d in df['cal_week']]


df['formatted_date'] = df.year.astype(str)  + df.cal_week.astype(str) + '0'
df['date'] = pd.to_datetime(df['formatted_date'], format='%Y%W%w')

# df['datetime'] = pd.to_datetime(df.year.astype(str) + '-' + 
#                                 df.cal_week.astype(str) + '-1' , format="%Y-%W-%w").dt.strftime('%Y-%W')
# -

import datetime
df['conv_date']= df.date.map(datetime.datetime.toordinal)


df



df.groupby('date').agg('sum')

# ## Data visualization
#

grpd_date = df.groupby('date').agg('sum') 
grpd_date = grpd_date.rename(columns={'counts':'nr_deaths'}) 
grpd_date

grpd_date.describe()

# +
upper_limit = grpd_date['nr_deaths'].quantile(0.99)
lower_limit = grpd_date['nr_deaths'].quantile(0.01)

new_df = grpd_date[(
    grpd_date['nr_deaths'] <= upper_limit) & (
    grpd_date['nr_deaths'] >= lower_limit)]
# -

new_df

grpd_date = new_df

fig = px.line(
    x=grpd_date.index,
    y=grpd_date.nr_deaths, 
    title='Deaths per year'
)
fig.show()

# +

len(x), len(y)
# -

grpd_date.index[0]

# +

fig = px.scatter(
    x=grpd_date.index,
    y=grpd_date.nr_deaths, 
    trendline="ols",
    trendline_color_override="red",
    opacity=.5,
    title='Deaths per year'
)
fig.show()

# regression params not available for lowess

# +

fig = px.scatter(
    x=grpd_date.index,
    y=grpd_date.nr_deaths, 
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


x=grpd_date.index.values


y=grpd_date.nr_deaths.values
x = np.arange(0, len(y))

# x = grpd_date.conv_date.values
x = x.reshape(-1, 1)
model = LinearRegression()
model.fit(x, y)




# x = grpd_date.index
# x = np.arange(0, len(y))

x_range_ordinal = np.linspace(x.min(), x.max(), len(y))


y_range = model.predict(x_range_ordinal.reshape(-1, 1))


len(x_range_ordinal), len(y_range)
                    
# reg.score(X, y)
# reg.coef_
# reg.intercept_                         
                          
# # Using a pipeline to automate the input transformation
# from sklearn.pipeline import Pipeline

# poly = PolynomialFeatures(degree)
# model = LinearRegression()
# pipeline = Pipeline(steps=[('t', poly), ('m', model)])

# linreg = pipeline.fit(X_train, y_train)
# y_predict2 = linreg.predict(X_predict)

# assert(np.array_equal(y_predict, y_predict2))


# +

fig = px.scatter(
    x=grpd_date.index,
    y=grpd_date.nr_deaths, 
    opacity=.5,
#     trendline='ols', trendline_color_override='darkblue',
    title='Deaths per year'
)


# x_range = pd.date_range(start=grpd_date.index[0],
#                   end=grpd_date.index[-1],
#                   periods=len(grpd_date.index.values))
fig.add_traces(
    go.Scatter(
        x=grpd_date.index, y=y_range, 
        name='Regression Fit'))


fig.show()


# +
from sklearn.preprocessing import PolynomialFeatures

poly_degree = 3


y = grpd_date.nr_deaths.values

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

# dx = np.diff(np.arange(0, len(y))),
# dy = np.diff(y_range_poly)
# slopes = dy/dx
# slopes = slopes[0]

# print('slps', len(slopes), slopes[100:200])


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
# -

slope_at_point[:50]

grpd_date.index

# +
# print(poly_reg_model.intercept_)
# print(poly_reg_model.coef_)

interc = poly_reg_model.intercept_
coeffs = poly_reg_model.coef_

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

grpd_date.index

y_value_at_point.flatten()#.shape, y_range_poly.shape

np.arange(0, len(y))

# +
fig = px.scatter(
    x=np.arange(0, len(y)),
    y=grpd_date.nr_deaths, 
    opacity=.5,
#     trendline='ols', trendline_color_override='darkblue',
    title='Deaths per year'
)


# x_range = pd.date_range(start=grpd_date.index[0],
#                   end=grpd_date.index[-1],
#                   periods=len(grpd_date.index.values))
fig.add_traces(
    go.Scatter(
        x=np.arange(0, len(y)), y=y_range_poly, 
        name='Polynomial regression Fit'))

fig.add_traces(
    go.Scatter(
        x=np.arange(0, len(y)), y=y_value_at_point, 
        name='Polynomial regression Fit 2'))




for pt in [1000]:
    slope_line(
        fig, 
        x= np.arange(0, len(y)), 
        y = y_value_at_point, 
        ind = pt)

# fig.update_layout(yaxis_range=[0,2000])

fig.show()

# -

len(grpd_date.index), len(grpd_date.nr_deaths)

# +
trace1 = go.Scatter(
                    x=grpd_date.index,
                    y=grpd_date.nr_deaths,
                    mode='lines',
                    line=dict(width=1.5))


frames=[
    dict(
        data=[
            dict(
                type = 'scatter',
                x=grpd_date.index[:k],
                y=grpd_date.nr_deaths[:k])]
    )
    for k in range(0, len(grpd_date))]

layout = go.Layout(width=700,
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




