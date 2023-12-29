import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns




st.set_page_config(
    page_title="Sentiment Analyzer", page_icon="📊", layout="wide"
)

st.title("Sentiment Analysis")
st.markdown("------------------------------------------------------------------------------------")

st.sidebar.markdown("Made with love using [streamlit](https://streamlit.io/)")
st.sidebar.image(
    "../images/sentiment-analysis.png"
)

data = pd.read_csv('data.csv')
col1, col2 = st.columns(2)
with col1:
    # sentiment_counts = data['sentiment_label'].value_counts()
    sentiment_counts = sns.countplot(data, x="sentiment_label", palette=["#7fcdbb", '#432371',"#FAAE7B" ])
    st.pyplot(sentiment_counts.get_figure())

with col2:
    piechart = data.sentiment_label.value_counts().plot(kind='pie')
    st.pyplot(piechart)



