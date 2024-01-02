import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
import seaborn as sns
from scipy import stats

# Define your color palette for the sentiment labels
colors = ["#7fcdbb", '#432371',"#FAAE7B" ]

st.set_page_config(
    page_title="Sentiment Analyzer", page_icon="📊", layout="wide"
)

st.title("Sentiment Analysis Dashboard 😊😐😡")
st.markdown("------------------------------------------------------------------------------------")

st.sidebar.markdown("Made with ❤️ using [streamlit](https://streamlit.io/)")
st.sidebar.image(
    "../images/sentiment-analysis.png"
)
data = pd.read_csv('data.csv')

# Show a quick summary of the dataset
st.sidebar.title('Dataset Overview')
st.sidebar.markdown(f"Total Reviews: {len(data)}")
st.sidebar.markdown(f"Avg. Sentiment Score: {data['sentiment_score'].mean():.2f}")




# Filter Sidebar
st.sidebar.title('Filters')
sentiment_label = st.sidebar.multiselect(
    "Choose Sentiment Labels",
    options=data['sentiment_label'].unique(),
    default=data['sentiment_label'].unique()
)

# Filtering data based on selection
if sentiment_label:
    data = data[data['sentiment_label'].isin(sentiment_label)]


#Include more contact or author information
st.sidebar.markdown('---')
# st.sidebar.markdown('👩‍💻 Developed by [Your Name](https://yourwebsite.com)')
st.sidebar.markdown('📢 Follow us on [Github](https://github.com/Milad-agdam) [LinkedIn](https://ir.linkedin.com/in/milad-gashangi-agdam-)')


col1, col2 = st.columns(2)


# Visualization functions
def create_treemap(data, path, values, color_discrete_sequence):
    fig = px.treemap(data, path=path, values=values, color_discrete_sequence=color_discrete_sequence)
    return fig

def create_pie(data, values, names, color_discrete_sequence):
    fig = px.pie(data, values=values, names=names, color_discrete_sequence=color_discrete_sequence)
    return fig


# Column 1: Sentiment Label Distribution and Treemap
with col1:
    st.subheader('Average Sentiment Score by Sentiment Label')
    sentiment_means = data.groupby('sentiment_label')['sentiment_score'].mean().reset_index()
    fig, ax = plt.subplots()
    sns.barplot(x='sentiment_label', y='sentiment_score', data=sentiment_means, palette=colors)
    ax.set_xlabel('Sentiment Label')
    ax.set_ylabel('Average Sentiment Score')
    st.pyplot(fig)
    
    st.subheader('Distribution of Sentiment Labels')
    sentiment_counts = data['sentiment_label'].value_counts().reset_index()
    sentiment_counts.columns = ['sentiment_label', 'counts']
    
    fig = create_treemap(sentiment_counts, path=['sentiment_label'], values='counts', color_discrete_sequence=colors)
    st.plotly_chart(fig, use_container_width=True)

# Column 2: Sentiment Proportion Pie Chart and Sentiment Scores Histogram
with col2:
    st.subheader('Sentiment Proportion')
    sentiment_counts = data['sentiment_label'].value_counts()
    
    fig = create_pie(data, values=sentiment_counts.values, names=sentiment_counts.index, color_discrete_sequence=colors)
    st.plotly_chart(fig)
    
    st.subheader('Distribution of Sentiment Scores')
    fig, ax = plt.subplots()
    sns.histplot(data, x='sentiment_score', bins=50, kde=False, color='#9e9e9e', edgecolor='#BB8FCE')
    ax.set_xlabel('Sentiment Score')
    ax.set_ylabel('Number of Reviews')
    st.pyplot(fig)
    

# Word Cloud for Positive Sentiment
st.subheader('Word Cloud for Positive Sentiment')
if 'positive' in sentiment_label:
    positive_comments = data["cleaned_review"][data['sentiment_label'] == "positive"]
    positive_text = " ".join(comment for comment in positive_comments)
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(background_color="white", max_words=50, stopwords=stopwords).generate(positive_text)
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

st.markdown("---")

