import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud


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
    # sentiment_counts = data['sentiment_label'].value_counts()
    sentiment_counts = sns.countplot(data, x="sentiment_label", palette=colors)
    st.pyplot(sentiment_counts.get_figure())
    st.subheader('Word Cloud for Positive Sentiment')
    positive_comments = data["cleaned_review"][data['sentiment_label']  == "positive"]
    positive_text = " ".join(comment for comment in positive_comments)
    wordcloud = WordCloud(background_color="white", ).generate(positive_text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')

with col2:
    # piechart = data.sentiment_label.value_counts().plot(kind='pie')
    # plt.axis('equal')
    # st.pyplot(piechart)
    # Display percentage of positive, negative and neutral sentiments
    st.subheader('Sentiment Proportion')
    positive = data['sentiment_label'][data['sentiment_label'] == "positive"].count()
    negative = data['sentiment_label'][data['sentiment_label'] == "negative"].count()
    neutral = data['sentiment_label'][data['sentiment_label'] == "neutral"].count()
    counts = [positive,negative, neutral]
    group = ['positive','negative', "neutral"]
    fig = px.pie(data['sentiment_label'], values=counts ,names=group, color=colors)
    st.plotly_chart(fig)
    st.subheader('Distribution of Sentiment Scores')
    fig, ax = plt.subplots()
    sns.histplot(data['sentiment_score'], bins=50, kde=False, ax=ax)
    ax.set_title('Distribution of Sentiment Scores')
    ax.set_xlabel('Sentiment Score')
    ax.set_ylabel('Number of Reviews')
    st.pyplot(fig)
    
# Set subheader for treemap
st.subheader('Treemap of Sentiment Distribution')

# Creating a treemap using Plotly
fig = px.treemap(sentiment_counts, path=['sentiment_label'], values='counts', title='Treemap of Sentiment Labels')

# Display Plotly treemap in Streamlit
st.plotly_chart(fig, use_container_width=True)
st.markdown("------------------------------------------------------------------------------------")

