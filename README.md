# Analysing Austrias deaths and their Covid-19 correlation

The data is used from **data.gv.at** which is Austrias official open data source. 

More about it [here](https://www.data.gv.at/infos/zielsetzung-data-gv-at/)


Central Notebook is [covid_analysis_charts.ipynb](covid_analysis_charts.ipynb)


--- 

# Published notebook on 

https://createdd.github.io/covid_analysis/covid_analysis_charts.html 

--- 
## Datasets
The used datasets are from:

general deaths: https://www.data.gv.at/katalog/dataset/stat_gestorbene-in-osterreich-ohne-auslandssterbefalle-ab-2000-nach-kalenderwoche/resource/b0c11b84-69d4-4d3a-9617-5307cc4edb73

covid deaths: https://www.data.gv.at/katalog/dataset/covid-19-zeitverlauf-der-gemeldeten-covid-19-zahlen-der-bundeslander-morgenmeldung/resource/24f56d99-e5cc-42e4-91fa-3e6a06b73064?view_id=89e19615-c7b3-445c-9924-e9dd6b8ace75

## Goal of the analysis
This project has multiple goals.

Work with Austrias open source data and see its ease of use
Plot interactively the development of deaths and covid deaths over time
Fit a (polynomial) regression through the data and calculate the slopes and given points to see the change of deaths rates



## Analysis of covid deaths

![](https://media.giphy.com/media/2sFU9wtlGAfcihhtlI/giphy.gif)


## Get Started

Install dependencies like

```shell
conda install --file requirements.txt
```

Run notebook like

```shell
jupyter notebook
```

## Streamlit App

To start the streamlit app, run the following in your terminal
```shell
streamlit run main.py
```

In case your environment cannot be found register your env as kernel like

See [here](https://www.python-engineer.com/posts/setup-jupyter-notebook-in-conda-environment/)

```shell
ipython kernel install --user --name=<any_name_for_kernel>
```

This will open a server, demonstrating 

![](https://media.giphy.com/media/AmHDlRuNlQ7oHoAcib/giphy.gif)

Furhter ideas to add for the streamlit app, [here](https://dataqoil.com/2022/02/20/creating-awesome-data-dashboard-with-plotly-in-streamlit/
)