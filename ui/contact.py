import requests
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_lottie import  st_lottie
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json

file = open('./ui/contact.json','rb')
cont = json.load(file)

def contact(df):
	st.header('Contact Us')

	for label, value in cont.items():
		st.text(label+": "+value)