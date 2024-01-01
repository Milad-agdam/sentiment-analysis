import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud
from wordcloud import ImageColorGenerator


# Define your color palette for the sentiment labels
colors = ["#7fcdbb", '#432371',"#FAAE7B" ]

st.set_page_config(
    page_title="Sentiment Analyzer", page_icon="📊", layout="wide"
)

st.title("Sentiment Analysis Dashboard 😊😐😡")
st.markdown("------------------------------------------------------------------------------------")

st.sidebar.markdown("Made with love using [streamlit](https://streamlit.io/)")
st.sidebar.image(
    "../images/sentiment-analysis.png"
)


data = pd.read_csv('data.csv')

col1, col2 = st.columns(2)

with col1:
    st.subheader('Distribution of Sentiment Labels')
    # Create the countplot
    sns.set_theme(style="whitegrid")
    sentiment_counts = sns.countplot(data, x="sentiment_label", palette=colors)
    st.pyplot(sentiment_counts.get_figure())
    
    # Set subheader for treemap
    sentiment_counts = data['sentiment_label'].value_counts().reset_index()
    sentiment_counts.columns = ['sentiment_label', 'counts']


    st.subheader('Treemap of Sentiment Distribution')

    # Creating a treemap using Plotly
    fig = px.treemap(sentiment_counts, path=['sentiment_label'], values='counts', color_discrete_sequence=colors)

    # Display Plotly treemap in Streamlit
    st.plotly_chart(fig, use_container_width=True)
    

with col2:
    st.subheader('Sentiment Proportion')
    positive = data['sentiment_label'][data['sentiment_label'] == "positive"].count()
    negative = data['sentiment_label'][data['sentiment_label'] == "negative"].count()
    neutral = data['sentiment_label'][data['sentiment_label'] == "neutral"].count()
    counts = [positive,negative, neutral]
    group = ['positive','negative', "neutral"]
    fig = px.pie(data['sentiment_label'], values=counts ,names=group, color_discrete_sequence=colors)
    st.plotly_chart(fig)
    
    #create a histogram
    st.subheader('Distribution of Sentiment Scores')
    fig, ax = plt.subplots()
    sns.histplot(data['sentiment_score'], bins=50, kde=False, ax=ax, color='#9e9e9e', edgecolor='#BB8FCE')
    # ax.set_title('Distribution of Sentiment Scores')
    ax.set_xlabel('Sentiment Score')
    ax.set_ylabel('Number of Reviews')
    st.pyplot(fig)
    

    # Create the wordcount
st.subheader('Word Cloud for Positive Sentiment')
positive_comments = data["cleaned_review"][data['sentiment_label']  == "positive"]
positive_text = " ".join(comment for comment in positive_comments)
# Add this line for debugging
wordcloud = WordCloud(background_color="white").generate(positive_text)
fig, ax = plt.subplots(figsize = (12, 12))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
st.pyplot(fig)
st.markdown("------------------------------------------------------------------------------------")

