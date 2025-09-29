import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("train.csv")

st.title("Projet de classification binaire Titanic")
st.sidebar.title("Sommaire")
pages=["Exploration", "DataVizualization", "Modélisation"]
page=st.sidebar.radio("Aller vers", pages)

if page == pages[0] : 
  st.write("### Introduction")

st.dataframe(df.head(10))

st.write("shape:", df.shape)
st.dataframe(df.describe())

if st.checkbox("Afficher les NA") :
  st.dataframe(df.isna().sum())

if page == pages[1] : 
  st.write("### DataVizualization")
