import requests
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_lottie import  st_lottie
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from ui.analysis import analysis
from ui.contact import contact
from ui.form import form
from ui.description import description

st.set_page_config(page_title='Stroke Prediction AI', page_icon=":tada:")

selected = option_menu(
    menu_title="Stroke Prediction AI", # required
    options=['Description','Form', 'Analysis', "Contact"],
    default_index = 0,
    orientation = 'horizontal',
    styles={
        "container": {"padding":"0!important", "background-color":"write",},
        "nav-link":{
            "font-size":"18px",
            "text-align":"left",
            "margin":"0px",
            },
        "nav-link-selected":{"background-color":"gray"}, 
            },
    
)  

df = pd.read_csv('data/data.csv')

if selected == 'Description':
	description(df)

if selected == 'Form':
	form(df)

if selected == "Analysis":
    analysis(df)

if selected == 'Contact':
	contact(df)