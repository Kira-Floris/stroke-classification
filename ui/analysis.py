import requests
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_lottie import  st_lottie
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def piechart(df, column):
    counts =list(df[column].value_counts().items())
    status =np.array([x[0] for x in counts])
    number =np.array([x[1] for x in counts])
    
    fig = go.Figure(
            go.Pie(
                labels = status,
                values = number,
                hoverinfo = 'label+percent',
                textinfo = 'value')
        )
    st.subheader('Visualization of '+column)
    st.plotly_chart(fig)


def filter(df):
	st.sidebar.header('Filter')
	gender = st.sidebar.multiselect('Select a Gender', options=list(df['gender'].unique()), default=list(df['gender'].unique()))
	age = st.sidebar.slider('Select age range', 0.0, float(df['age'].max()), (0.0,float(df['age'].max())), key='age')
	hypertension = st.sidebar.multiselect('Hypertension: 1 for true, 0 for false', options=list(df['hypertension'].unique()), default=list(df['hypertension'].unique()))
	heart_disease = st.sidebar.multiselect('heart_disease: 1 for true, 0 for false', options=list(df['heart_disease'].unique()), default=list(df['heart_disease'].unique()))
	ever_married = st.sidebar.multiselect('Select marital status', options=list(df['ever_married'].unique()), default=list(df['ever_married'].unique()))
	work_type = st.sidebar.multiselect('Select your work type', options=list(df['work_type'].unique()), default=list(df['work_type'].unique()))
	residence_type = st.sidebar.multiselect('Select your residence type', options=list(df['Residence_type'].unique()), default=list(df['Residence_type'].unique()))
	avg_glucose_level = st.sidebar.slider('Select average glucose range', 0.0, float(df['avg_glucose_level'].max()), (0.0,float(df['avg_glucose_level'].max())), key='avg_glucose_level')
	bmi = st.sidebar.slider('Select bmi range', 0.0, float(df['bmi'].max()), (0.0,float(df['bmi'].max())), key='bmi')
	smoking_status = st.sidebar.multiselect('Select your smoking status', options=list(df['smoking_status'].unique()), default=list(df['smoking_status'].unique()))

	df = df[df['gender'].isin(gender)]
	df = df[df['hypertension'].isin(hypertension)]
	df = df[df['heart_disease'].isin(heart_disease)]
	df = df[df['ever_married'].isin(ever_married)]
	df = df[df['work_type'].isin(work_type)]
	df = df[df['Residence_type'].isin(residence_type)]
	df = df[df['smoking_status'].isin(smoking_status)]

	df = df[df['age'].between(age[0], age[1], inclusive=False)]
	df = df[df['avg_glucose_level'].between(avg_glucose_level[0], avg_glucose_level[1], inclusive=False)]
	df = df[df['bmi'].between(bmi[0], bmi[1], inclusive=False)]

	st.dataframe(df)

	piechart(df, 'gender')
	piechart(df, 'hypertension')
	piechart(df, 'heart_disease')
	piechart(df, 'ever_married')
	piechart(df, 'work_type')
	piechart(df, 'Residence_type')
	piechart(df, 'smoking_status')



def analysis(df):
	st.header('Analysis')
	filter(df)

	return 