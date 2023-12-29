import nltk
import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
import emoji
import matplotlib.pyplot as plt


df = pd.read_csv('output.csv')

# detect rows with missing reviews
df.isnull().sum()

def clean_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove whitespaces
    text = text.strip()
    return text
    
def tokenize(text):
    # split the text into individual words
    return text.split()

stop_words = set(stopwords.words('english'))

def remove_stopwords(tokens):
    #Remove common words that usually don’t carry important meaning.
    return [word for word in tokens if not word in stop_words]

lemmatizer = WordNetLemmatizer()

def lemmatize_tokens(tokens):
    return [lemmatizer.lemmatize(token) for token in tokens]

def rejoin_tokens(tokens):
    return ' '.join(tokens)

def remove_emojis(text):
    # Remove emojis from the text
    cleaned_text = emoji.demojize(text)
    return cleaned_text

positive_words = set(["legend","good", "excellent", "awesome", "amazing", "fantastic", "finest", "fantastic", "quick", "beautifull"])

# Function to adjust sentiment based on positive words
def adjust_sentiment_based_on_positive_words(text, sentiment_score):
    # Check if any positive words are present in the text
    if any(word in text for word in positive_words):
        # Increase the sentiment score (adjust as needed)
        sentiment_score += 0.1
    return sentiment_score
    # Remove emojis from the text
    cleaned_text = emoji.demojize(text)
    return cleaned_text

def preprocess_pipeline(text):
    # Chain your preprocessing functions
    text = clean_text(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_tokens(tokens)
    text = rejoin_tokens(tokens)
    text = remove_emojis(text)
    return text

df['cleaned_review'] = df['comment'].apply(preprocess_pipeline)

# Initialize the SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Analyze sentiment and create a new column
df['sentiment_score'] = df.apply(lambda row: sia.polarity_scores(row['cleaned_review'])['compound'], axis=1)

# Adjust sentiment based on positive words
df['sentiment_score'] = df.apply(lambda row: adjust_sentiment_based_on_positive_words(row['cleaned_review'], row['sentiment_score']), axis=1)
# Define a function to label sentiments
def label_sentiment(score):
    if score >= 0.05:
        return 'positive'
    elif score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

# Apply the function to create a new column with sentiment labels
df['sentiment_label'] = df['sentiment_score'].apply(label_sentiment)