import requests
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_lottie import  st_lottie
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def description(df):
	file = open('./ui/description.docx', 'r')
	st.markdown(file.read())