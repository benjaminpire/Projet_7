import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import requests
import json
import seaborn as sns
import lime.lime_tabular
import re
import plotly.graph_objects as go



# Permettre de visualiser le score et l’interprétation de ce score pour chaque client de façon intelligible pour une personne non experte en data science.


# Permettre de visualiser des informations descriptives relatives à un client (via un système de filtre).


# Permettre de comparer les informations descriptives relatives à un client à l’ensemble des clients ou à un groupe de clients similaires.
    
    
# diagramme en jauge de chez plotly 
    
    
        
def request_prediction(model_uri, data):
    headers = {"Content-Type": "application/json"}

    data_json = {'client_id': data}

    response = requests.post(headers=headers, url=model_uri, json=data_json)

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))

    return response.json()


def main():
    #get the data frames
    df = pd.read_csv( 'X_test.csv')
    X_train = pd.read_csv('X_train.csv')
    y_train = pd.read_csv('y_train.csv')
    #link to the url
    MLFLOW_URI = 'http://127.0.0.1:8000/predict'
    
    
    #create 2 columns of display : 
    container_corp = st.container()
    with container_corp:
        col1, col2 = st.columns([2,3])
    
    # column 1
    with col1:

        # selection of the client 
        client_select = st.selectbox(
                "Select **the ID** of the client",
               df["SK_ID_CURR"])
        variable = st.multiselect('Selectionnez quelques variables',
                                  df.columns,
                                  "DAYS_REGISTRATION")
        predict_btn = st.button('Prédire')
        # prediction 
        if predict_btn:
            pred = None
            pred = request_prediction(MLFLOW_URI,client_select)
            st.markdown("Le client selectioné  est " + pred["class"] ) 
            st.markdown("Avec une probailité de "  + str(pred["proba"]))
            #st.markdown("Voicis maintenant les résultats de l'interprétabilité locale "  + #str(pred["inter"]))

        # selectionner quelques variables
        if predict_btn:
            for i in range(len(variable)):
                st.metric(label=variable[i],
                          value=df.loc[df["SK_ID_CURR"] == client_select, variable[i]])
        
        
        
    # column 2
    with col2:
        if predict_btn:
            for i in range(len(variable)):
                st.line_chart(df, x="SK_ID_CURR", y=variable[i])
                
                
                
if __name__ == '__main__':
    main()
