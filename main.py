import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


final_df = pd.read_excel('data/cleaned_data.xlsx')


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

st.plotly_chart(fig)