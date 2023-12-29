import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load your data into a Pandas DataFrame
# df = pd.read_csv('data.csv')


st.set_page_config(
    page_title="Sentiment Analyzer", page_icon="📊", layout="wide"
)

st.title("Sentiment Analysis")
st.markdown("------------------------------------------------------------------------------------")


filename = st.sidebar.file_uploader("Upload reviews data:", type=("csv", "xlsx"))

if filename is not None:
    data = pd.read_csv(filename)
    col1, col2 = st.columns(2)
    with col1:
        sentiment_counts = data['sentiment_label'].value_counts()
        st.bar_chart(sentiment_counts)
    