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
import streamlit.components.v1 as components
import plotly.express as px
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
    
    tab1, tab2, tab3 = st.tabs(["Informations client", "Comparaison avec d'autres clients", "Score interprétation"])

    
    with tab1:
        # selection of the client 
        client_select = st.selectbox(
                "Select **the ID** of the client",
               df["SK_ID_CURR"])
        
        col_list = ["SK_ID_CURR", "DAYS_BIRTH","CNT_CHILDREN", "FLAG_EMP_PHONE", "FLAG_EMAIL", "NAME_EDUCATION_TYPE_Highereducation","CODE_GENDER", "AMT_CREDIT", "AMT_INCOME_TOTAL"]
        st.dataframe(
                df.loc[df["SK_ID_CURR"] == client_select, col_list],
                column_config={
                    "SK_ID_CURR": "ID du client",
                    "DAYS_BIRTH": "Années de naissance",
                    "CNT_CHILDREN": "numbers of children",
                    "FLAG_EMP_PHONE": "Personal phone",
                    "FLAG_EMAIL": "Mail",
                    "NAME_EDUCATION_TYPE_Highereducation": "Higher education",
                    "CODE_GENDER": "genre",
                    "AMT_CREDIT": "Montant du crédit",
                    "AMT_INCOME_TOTAL": "Total des entrans"},
                height = 75)
        

        predict_btn = st.button('Prédire')

        col1, col2 = st.columns(2)
        if predict_btn:
            pred = None
            pred = request_prediction(MLFLOW_URI,client_select)
            st.metric("Le client selectioné  est", value=pred["class"])
        

    with tab2:
        variable = st.multiselect('Selectionnez quelques variables',
                                  df.columns) 
        col_binaire = df.loc[:,[col for col in df.columns if 'FLAG' in col]].columns.tolist()
        col_binaire.append('CODE_GENDER')
        col1, col2 = st.columns([1, 4])
        for i in range(len(variable)):
            df["str_SK_ID_CURR"] = df["SK_ID_CURR"].astype(str)
            client_value = df.loc[df["SK_ID_CURR"] == client_select, variable[i]]
            if variable[i] in col_binaire: 
                with col1:
                    st.metric(label=variable[i],
                              value=client_value)
            else: 
                with col2:
                    colors = {client_select: "red"}
                    color_discrete_map = {c: colors.get(c, "blue") for c in df["SK_ID_CURR"]}
                    fig = px.strip(df, x = variable[i])
                    fig2 = px.strip(df.loc[df["SK_ID_CURR"] == client_select], x = variable[i], color_discrete_sequence = ["red"]).data
                    fig.add_traces(fig2)

                    
                    st.plotly_chart(fig)

        
        
    with tab3:
        col1, col2 = st.columns(2)

        # selectionner quelques variables
        if predict_btn:
            with col1:
                pred = None
                pred = request_prediction(MLFLOW_URI,client_select)
                st.metric("Le client selectioné  est", value=pred["class"])
            with col2:
                st.metric("Avec une probailité de", value=pred["proba"])
            
            #components.html(pred["inter"], height=800)        


if __name__ == '__main__':
    main()
