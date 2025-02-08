from operator import index
import streamlit as st
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pycaret.datasets import get_data
from sklearn.metrics import confusion_matrix
from pycaret.classification import *
from pycaret.anomaly import *
from pycaret.regression import setup, compare_models, pull, save_model, load_model,evaluate_model,plot_model
from pycaret.regression import *
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
#from sklearn.preprocessing import LabelEncoder
import os 
import sys
sys.setrecursionlimit(10000000)

option=st.selectbox("Select an option:", ["Regression","Classification"])
#st.button("Select the option",option)
with st.sidebar:
        st.image("https://www.onepointltd.com/wp-content/uploads/2019/12/ONE-POINT-01-1.png")
        st.title("Auto ML App")
        choice=st.radio("Navigation",["Upload","Profile","Predict","Download"])
        st.info("Automating the end-to-end process of applying machine learning algorithms on real-world problems")

    #option=st.selectbox("Select an option:", ["Regression","Classification"])


if option=='Regression':
    if os.path.exists('./dataset.csv'): 
         df = pd.read_csv('dataset.csv', index_col=None)

    if choice=="Upload":
        st.image("https://cdn.educba.com/academy/wp-content/uploads/2021/09/DataSet-Type.jpg.webp")
        st.title("Please Upload Your Dataset")
        file=st.file_uploader("Dataset Must Be in CSV Format")
        if file: 
            df = pd.read_csv(file, index_col=None)
            df.to_csv("dataset.csv", index=None)
            st.dataframe(df)

    if choice == "Profile": 
        st.title("Exploratory Data Analysis")
        profile_df =ProfileReport(df, minimal=True)
        st_profile_report(profile_df)
        report_file_path = "profile_report.pdf"
        profile_df.to_file(report_file_path)
        st.download_button(label='Download Profile',data=report_file_path,file_name="profile_report.pdf")

    if choice=="Predict": 
        choices = st.selectbox("Select an option:", ["Preprocess","Run Column"])
        if choices== "Preprocess":
            df = pd.read_csv('dataset.csv')
            df_preprocessed = df.fillna(0)
            df_preprocessed = df_preprocessed.where(pd.notna(df), 1)
            df_preprocessed.to_csv('preprocessed.csv',index=False)
            if st.button("Perprocess"):
                st.success("Preprocessing completed!")   
                st.dataframe(df)

                    
        elif choices == "Run Column":
            df=pd.read_csv('preprocessed.csv')
            ch_target = st.selectbox("Choose the Target", df.columns)
            if st.button("Run Modelling"): 
                chunk_size = 10000  # Adjust the chunk size based on your available memory
                setup(df, target=ch_target, verbose = False)
                setup_df = pull()
                st.dataframe(setup_df)
                best_model = compare_models()
                compare_df = pull()
                save_model(best_model, 'best_model')
                st.dataframe(compare_df)
                    
                    
                    
            

    if choice == "Download": 
        with open('best_model.pkl', 'rb') as f: 
            st.download_button('Download Model', f, file_name="best_model.pkl")

if option=='Classification':
    from pycaret.classification import *
    if os.path.exists('./dataset.csv'): 
        df = pd.read_csv('dataset.csv', index_col=None)

    if choice=="Upload":
        st.image("https://cdn.educba.com/academy/wp-content/uploads/2021/09/DataSet-Type.jpg.webp")
        st.title("Please Upload Your Dataset")
        file=st.file_uploader("Dataset Must Be in CSV Format")
        if file: 
            df = pd.read_csv(file, index_col=None)
            df.to_csv("dataset.csv", index=None)
            st.dataframe(df)

    if choice == "Profile": 
        st.title("Exploratory Data Analysis")
        profile_df =ProfileReport(df, minimal=True)
        st_profile_report(profile_df)
        report_file_path = "profile_report.pdf"
        profile_df.to_file(report_file_path)
        st.download_button(label='Download Profile',data=report_file_path,file_name="profile_report.pdf")

    if choice=="Predict": 
        choices = st.selectbox("Select an option:", ["Preprocess","Run Column"])
        if choices== "Preprocess":
            df = pd.read_csv('dataset.csv')
            df_preprocessed = df.fillna(0)
            df_preprocessed = df_preprocessed.where(pd.notna(df), 1)
            df_preprocessed.to_csv('preprocessed.csv',index=False)
            if st.button("Perprocess"):
                st.success("Preprocessing completed!")   
                st.dataframe(df)

                    
        elif choices == "Run Column":
            df = pd.read_csv('preprocessed.csv')
            #cat_features = list(df.columns)
            ch_target = st.selectbox("Choose the Target", df.columns)
            if st.button("Run Modelling"): 
                chunk_size = 10000  # Adjust the chunk size based on your available memory
                setup(df,target=ch_target)
                setup_df=pull()
                st.dataframe(setup_df)
                best_model=compare_models()
                compare_df=pull()
                save_model(best_model, 'best_model')
                st.dataframe(compare_df)

            

    if choice == "Download": 
        with open('best_model.pkl', 'rb') as f: 
            st.download_button('Download Model', f, file_name="best_model.pkl")









#st.write("HELLO AUTO ML")