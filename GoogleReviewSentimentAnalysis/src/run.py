import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

colors = ["#7fcdbb", '#432371',"#FAAE7B" ]

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
    sentiment_counts = sns.countplot(data, x="sentiment_label", palette=colors)
    st.pyplot(sentiment_counts.get_figure())

with col2:
    # piechart = data.sentiment_label.value_counts().plot(kind='pie')
    # plt.axis('equal')
    # st.pyplot(piechart)
    # Display percentage of positive, negative and neutral sentiments
    positive = data['sentiment_label'][data['sentiment_label'] == "positive"].count()
    negative = data['sentiment_label'][data['sentiment_label'] == "negative"].count()
    neutral = data['sentiment_label'][data['sentiment_label'] == "neutral"].count()
    counts = [positive,negative, neutral]
    group = ['positive','negative', "neutral"]
    fig = px.pie(data['sentiment_label'], values=counts ,names=group, color=colors)
    st.plotly_chart(fig)
    
    
st.markdown("------------------------------------------------------------------------------------")

# st.components.html("http://marketing.mealzo.co.uk/Reports/powerbi/google-bussines")
st.markdown("""
    <iframe width="600" height="606" src="http://marketing.mealzo.co.uk/Reports/powerbi/google-bussines" frameborder="0" style="border:0" allowfullscreen></iframe>
    """, unsafe_allow_html=True)
