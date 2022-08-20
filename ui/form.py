import requests
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_lottie import  st_lottie
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import json

# models
dt = pickle.load(open('./models/dt.pkl', 'rb'))
gnb = pickle.load(open('./models/gnb.pkl', 'rb'))
knn = pickle.load(open('./models/knn.pkl', 'rb'))
lr = pickle.load(open('./models/lr.pkl', 'rb'))
svm = pickle.load(open('./models/svm.pkl', 'rb'))
rfc = pickle.load(open('./models/rfc.pkl', 'rb'))

file = open('./results/models.json','rb')
results = json.load(file)

def labelize(df, column, value):
	unique = list(df[column].unique())
	unique.sort()
	return unique.index(value)

def form(df):
	st.header('Results of our models')
	st.sidebar.title("models results")
	st.sidebar.json(results)
	st.title('Form for AI testing')
		
	inputs = dict()
	labels = dict()

	encode_col = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

	with st.form("ai form"):
		gender = st.selectbox('gender', set(df['gender']))
		age = st.number_input('age')
		hypertension = st.selectbox('hypertension 1 for true 0 for false', set(df['hypertension'])) 
		heart_disease = st.selectbox('heart_disease 1 for true 0 for false', set(df['heart_disease']))
		ever_married = st.selectbox('marital status', set(df['ever_married']))
		work_type = st.selectbox('work type', set(df['work_type']))
		residence_type = st.selectbox('residence type', set(df['Residence_type']))
		avg_glucose_level = st.number_input('average glucose level')
		bmi = st.number_input('bmi')
		smoking_status = st.selectbox('smoking status', set(df['smoking_status']))

	    # submition form
		submitted = st.form_submit_button("Submit")
		if submitted:
			inputs['gender'] = labelize(df, 'gender', gender)
			inputs['age'] = age
			inputs['hypertension'] = hypertension
			inputs['heart_disease'] = heart_disease
			inputs['ever_married'] = labelize(df, 'ever_married', ever_married)
			inputs['work_type'] = labelize(df, 'work_type', work_type)
			inputs['residence_type'] = labelize(df, 'Residence_type', residence_type)
			inputs['avg_glucose_level'] = avg_glucose_level
			inputs['bmi'] = bmi
			inputs['smoking_status'] = labelize(df, 'smoking_status', smoking_status)

		    # predicting for the model
			
			dt_ = dt.predict([[inputs['gender'],inputs['age'],inputs['hypertension'],inputs['heart_disease'],inputs['ever_married'],inputs['work_type'],inputs['residence_type'],inputs['avg_glucose_level'],inputs['bmi'],inputs['smoking_status']]])
			rfc_ = rfc.predict([[inputs['gender'],inputs['age'],inputs['hypertension'],inputs['heart_disease'],inputs['ever_married'],inputs['work_type'],inputs['residence_type'],inputs['avg_glucose_level'],inputs['bmi'],inputs['smoking_status']]])
			gnb_ = gnb.predict([[inputs['gender'],inputs['age'],inputs['hypertension'],inputs['heart_disease'],inputs['ever_married'],inputs['work_type'],inputs['residence_type'],inputs['avg_glucose_level'],inputs['bmi'],inputs['smoking_status']]])
			knn_ = knn.predict([[inputs['gender'],inputs['age'],inputs['hypertension'],inputs['heart_disease'],inputs['ever_married'],inputs['work_type'],inputs['residence_type'],inputs['avg_glucose_level'],inputs['bmi'],inputs['smoking_status']]])
			lr_ = lr.predict([[inputs['gender'],inputs['age'],inputs['hypertension'],inputs['heart_disease'],inputs['ever_married'],inputs['work_type'],inputs['residence_type'],inputs['avg_glucose_level'],inputs['bmi'],inputs['smoking_status']]])
			svm_ = svm.predict([[inputs['gender'],inputs['age'],inputs['hypertension'],inputs['heart_disease'],inputs['ever_married'],inputs['work_type'],inputs['residence_type'],inputs['avg_glucose_level'],inputs['bmi'],inputs['smoking_status']]])
			
			st.subheader('Predicted Classifications')

			st.text('decision tree: '+str(dt_[0]))
			st.text('random forest classifier: '+str(rfc_[0]))
			st.text('gnb: '+str(gnb_[0]))
			st.text('knn: '+str(knn_[0]))
			st.text('linear regression: '+str(lr_[0]))
			st.text('svm: '+str(svm_[0]))