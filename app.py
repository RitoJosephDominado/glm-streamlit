


import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

import matplotlib.ticker as ticker
import statsmodels.api as sm
import statsmodels.formula.api as smf

import plotly.express as px


from streamlit_extras import stylable_container
# from streamlit.extras import e
from eda import create_missing_values_barplot
from eda import create_damage_histogram
from eda import create_age_plotly
from eda import create_damage_below_15000_histogram
from eda import create_age_histogram
from eda import plot_categorical_barplots
from eda import create_age_plotly
from eda import create_damage_plotly

from modeling import prepare_data
from modeling import create_premium_df
from modeling import print_error_metrics
box_css = """
    {
        border: 1px solid rgba(49, 51, 63, 0.2);
        border-radius: 0.5rem;
        padding: calc(1em - 1px);
        background-color: rgba(170, 255, 170, 0.8);
    }
    """

box_css = """
{
  /* Set the background color to a light pastel green */
  background-color: #e0f2e0; /* A soft, calming green */
  
  /* Add some internal padding for spacing */
  padding: 2.5rem; /* Large padding for a clean, spacious feel */
  
  /* Give it rounded corners */
  border-radius: 1.5rem; /* Generous rounding for a soft look */
  
  /* Use a box shadow for a 'floating' effect */
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1), /* Main shadow */
              0 1px 3px rgba(0, 0, 0, 0.08); /* Secondary, lighter shadow */
  
  /* Add a very subtle border */
  border: 1px solid rgba(0, 0, 0, 0.05); /* A thin, barely-there border */
  
  /* Center the box and give it a max width for responsiveness */
  margin: 2rem auto;
  max-width: 900px;
}
"""

orange_box_css = """
  /* Set the background color to a light pastel orange */
  background-color: #f2e0d3; /* A soft, calming orange */
  
  /* Add some internal padding for spacing */
  padding: 2.5rem; /* Large padding for a clean, spacious feel */
  
  /* Give it rounded corners */
  border-radius: 1.5rem; /* Generous rounding for a soft look */
  
  /* Use a box shadow for a 'floating' effect */
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1), /* Main shadow */
              0 1px 3px rgba(0, 0, 0, 0.08); /* Secondary, lighter shadow */
  
  /* Add a very subtle border */
  border: 1px solid rgba(0, 0, 0, 0.05); /* A thin, barely-there border */
  
  /* Center the box and give it a max width for responsiveness */
  margin: 2rem auto;
  max-width: 900px;
}
"""

st.set_page_config(layout="wide")
df = pd.read_csv('AutoBI_output.csv')
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

with data_tab:
    st.header('Technical Exam by Rito Dominado')
    st.image('car_crash_vecteezy.jpg',width=300)
    with stylable_container(
        key="container_with_border",
        css_styles="""
            {
                border: 1px solid rgba(49, 51, 63, 0.2);
                border-radius: 0.5rem;
                padding: calc(1em - 1px)
            }
            """,    
        ):
        
        st.dataframe(df)
with eda_tab:
    st.header('Exploratory Data Analysis 📊')
    with stylable_container(key='eda_container', css_styles=box_css):
        st.subheader('Missing Values')
        st.write('This bar plot shows the percentage of missing values in each column.')
        create_missing_values_barplot(df)

    numeric_eda_cols = st.columns([0.5, 0.5])
    # categorical_eda_cols = st.columns([0.2, 0.2, 0.2, 0.2, 0.2])
    categorical_eda_1_cols = st.columns([0.333, 0.333, 0.333])
    categorical_eda_2_cols = st.columns([0.6, 0.4])
    with numeric_eda_cols[0]:
        with stylable_container(key='damage_container', css_styles=box_css):
            st.subheader('Damage (ACTUALDAMAGE)')
            st.write('')
            # fig = create_damage_plotly(df)
            fig = px.histogram(df, x='ACTUALDAMAGE', nbins=20, title='Distribution of Damages')
            fig.update_traces(marker_line_width=1, marker_line_color="black")
            fig.update_layout(xaxis_title='Damage', yaxis_title='Count')
            st.plotly_chart(fig, key='damage_plotly')
    with numeric_eda_cols[1]:
        with stylable_container(key='age_container', css_styles=box_css):
            st.subheader('Age (INSAGE)')
            st.write('')
            st.html('''<ul>
            <li>This is the only numerical predictor in the dataset./li>
            <li>This has 14% of its values missing, the most of any column.</li>
            </ul>''')
            # st.write('This is the only numerical predictor in the dataset.')
            # st.write('This has 14% of its values missing, the most of any column.')
            fig = px.histogram(df, x='INSAGE', nbins=20, title='Distribution of Ages')
            fig.update_traces(marker_line_width=1, marker_line_color="black")
            fig.update_layout(xaxis_title='Age', yaxis_title='Count')
            st.plotly_chart(fig, key='age_plotly')
    
    with categorical_eda_1_cols[0]:
        fig = px.bar(df, x='VEHICLE_TYPE', title='Vehicle Types', color='VEHICLE_TYPE',color_discrete_sequence=px.colors.qualitative.Pastel)
        with stylable_container(key='vehicle_type_container', css_styles=orange_box_css):
            st.header('Vehicle Type')
            st.write(' ')
            st.write('It appears that the customers with small cars end up with higher damagess than those with big cars.')
            st.plotly_chart(fig, key='vehicle_type_plotly')
        
    with categorical_eda_1_cols[1]:
        with stylable_container(key='gender_container', css_styles=orange_box_css):
            st.subheader('Gender')
            fig = px.bar(df, x='GENDER', title='Genders', color='GENDER',color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig,key='gender_plotly')
    with categorical_eda_1_cols[2]:
        with stylable_container(key='prevclm_container', css_styles=box_css):
            st.subheader('Previous Claims (PREVCLM)')
            st.write(' ')
            fig = px.bar(df, x='PREVCLM', title='Previous Claims Options',color='PREVCLM',color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig, key='prevclm_plotly')

    with categorical_eda_2_cols[0]:
        with stylable_container(key='marital_status_container', css_styles=box_css):
            st.subheader('Marital Status')
            st.write(' ')
            fig = px.bar(df, x='MARITAL_STATUS', title='Marital Statuses', color='MARITAL_STATUS', color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig, key='marital_status_plotly')

    with categorical_eda_2_cols[1]:
        with stylable_container(key='seatbelt_container', css_styles=box_css):
            st.subheader('Seatbelt Usage')
            st.write(' ')
            st.write('The vast majority of customers used a seatbelts during the accident.')
            
            fig = px.bar(df, x='SEATBELT', title='Seatbelt Options',color='SEATBELT',color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig, key='seatbelt_plotly')

with model_tab:
    st.header('Modeling  📈')
    st.selectbox('Select Imputation Method', options=['KNN', 'Simple'], key='imputation_method')
    model = sm.OLS(train_target_series, prepared_feature_train_df).fit()
    st.write('Model Summary')    


with final_premium_tab:
    st.header('Final Premium Calculation')
    st.subheader('Note that the values here depend on the model used in the Modeling tab.')
    premium_df = create_premium_df(prepared_feature_test_df, test_target_series, model.predict(prepared_feature_test_df))
    st.dataframe(premium_df)

