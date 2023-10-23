import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import requests
import json
import seaborn as sns
import lime.lime_tabular
import re
import webbrowser



def main():
    
    
    #get the data frames
    df = pd.read_csv( 'X_test.csv')
    X_train = pd.read_csv('X_train.csv')
    y_train = pd.read_csv('y_train.csv')
    #link to the url
    MLFLOW_URI = 'http://127.0.0.1:8000/predict'
    
    
    ##### titles 
    container_title = st.container()
    with container_title:
        st.title("Dashboard de scoring crédit" ) 
    
    
    # selection of the client 
    client_select = st.selectbox(
            "Select **the ID** of the client",
            df["SK_ID_CURR"])
    predict_btn = st.button('Prédire')

        
        
        
        
#        st.markdown("[Result](https://streamlit.io)")


if __name__ == '__main__':
    main()
