


import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Normalizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from scipy.stats import chi2_contingency

import matplotlib.ticker as ticker
import statsmodels.api as sm
import statsmodels.formula.api as smf

from eda import create_missing_values_barplot
from eda import create_damage_histogram
from eda import *
from eda import create_damage_below_15000_histogram
from eda import create_age_histogram
from eda import plot_categorical_barplots
from eda import create_age_plotly

from modeling import prepare_data

st.set_page_config(layout="wide")
df = pd.read_excel('AutoBI.xlsx',sheet_name='Output')
feature_df = df.loc[:, ['INSAGE', 'VEHICLE_TYPE', 'GENDER', 'MARITAL_STATUS', 'PREVCLM', 'SEATBELT']]
target_series = df.loc[:, 'LOSS']
train_feature_df, test_feature_df, train_target_series, test_target_series = train_test_split(feature_df, target_series, test_size=0.3, random_state=123)

prepared_feature_train_df = prepare_data(train_feature_df)
prepared_feature_test_df = prepare_data(test_feature_df)

def create_glm_model(target_series, prepared_feature_df, model_type='OLS'):
    if model_type == 'OLS':
        model = sm.OLS(target_series, prepared_feature_df).fit()
    elif model_type == 'Gamma':
        model = sm.GLM(target_series, prepared_feature_df, family=sm.families.Gamma(link=sm.families.links.Log())).fit()
    elif model_type == 'Tweedie':
        model = sm.GLM(target_series, prepared_feature_df, family=sm.families.Tweedie(var_power=1.5, link=sm.families.links.Log())).fit()
    return model
data_tab, eda_tab, model_tab, final_premium_tab = st.tabs(['Data', 'EDA', 'OLS Model', 'Final Premium'])
# st.title('Technical Exam')
st.header('Technical Exam by Rito Dominado')
with data_tab:
    st.write('Data Overview')
    st.dataframe(df)
    # pivot_table(df,height=50,use_container_width=True, key="streamlit_pivottable")
with eda_tab:
    st.write('Exploratory Data Analysis')
    fig = create_damage_plotly(df)
    st.subheader('Damage (ACTUALDAMAGE) Distribution')
    numeric_eda_cols = st.columns([0.5, 0.5])
    categorical_eda_cols = st.columns([0.2, 0.2, 0.2, 0.2, 0.2])
    with numeric_eda_cols[0]:
        fig = px.histogram(df, x='ACTUALDAMAGE', nbins=20, title='Distribution of Damages')
        fig.update_traces(marker_line_width=1, marker_line_color="black")
        fig.update_layout(xaxis_title='Damage', yaxis_title='Count')
        st.plotly_chart(fig, key='damage_plotly')
    with numeric_eda_cols[1]:
        st.subheader('Numeric EDA')
        st.plotly_chart(create_damage_plotly(df))
    
    with categorical_eda_cols[0]:
        fig = px.bar(df, x='VEHICLE_TYPE', title='Vehicle Type Distribution')
        st.plotly_chart(fig, key='vehicle_type_plotly')
    with categorical_eda_cols[1]:
        fig = px.bar(df, x='GENDER', title='Vehicle Type Distribution')
        st.plotly_chart(fig,key='gender_plotly')
    with categorical_eda_cols[2]:
        fig = px.bar(df, x='MARITAL_STATUS', title='Vehicle Type Distribution')
        st.plotly_chart(fig, key='marital_status_plotly')
    # st.plotly_chart(fig)
    # st.plotly_chart(create_age_plotly(df))



# model_type = st.selectbox('Select Model Type', options=['OLS', 'Tweedie', 'Gamma'], key='model_type')
# model = create_glm_model(train_target_series, train_feature_df, model_type=model_type)
with model_tab:
    st.write('Ordinary Least Squares Model')
    st.selectbox('Select Imputation Method', options=['KNN', 'Simple'], key='imputation_method')
    model = sm.OLS(train_target_series, prepared_feature_train_df).fit()
    # model = create_glm_model(train_target_series, train_feature_df, model_type=st.session_state.model_type)
    
    st.write('Model Summary')
    # model_summary = model.summary()
    # st.write(model_summary.as_text())
    # st.write(model.summary())
    # st.dataframe(model_summary.tables[0])
    # st.dataframe(model_summary.tables[1])
    
    # st.html(model_summary.tables[0].as_html())
    # st.html(model_summary.tables[1].as_html())

    

with final_premium_tab:
    st.write('Final Premium Calculation')
    st.write('pivot should be above')

