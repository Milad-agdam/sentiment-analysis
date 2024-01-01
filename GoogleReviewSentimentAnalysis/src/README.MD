# Sentiment Analyzer and Analysis Dashboard

## Overview

This project comprises two main components: a sentiment analysis script and a sentiment analysis dashboard

## Sentiment Analysis Script

The script, written in Python, performs sentiment analysis on a dataset and exports the processed data to a CSV file. It utilizes the NLTK library for natural language processing, removing stopwords, lemmatizing, and adjusting sentiment scores based on positive words. The sentiment scores are then converted into sentiment labels (positive, negative, neutral).

## Requirements

- nltk==3.8.1
- pandas==1.3.3
- emoji==1.6.0

## Usage

1. Ensure the required libraries are installed using the provided requirements.txt file.
2. Place the dataset in a CSV file named `output.csv` in the same directory as the script.
3. Run the script to generate a processed dataset named data.csv.

# Sentiment Analysis Dashboard

The dashboard, built using Streamlit, visualizes the sentiment distribution through various plots such as countplots, treemaps, pie charts, histograms, and word clouds. It reads the processed dataset (data.csv) and provides an interactive interface for exploring sentiment patterns.

## Requirements
- nltk==3.8.1
- numpy==1.19.5
- wordcloud==1.9.3
- streamlit==1.14.0
- seaborn==0.11.2
- pandas==1.3.3
- matplotlib==3.4.3
- plotly==5.3.1

## Usage
1. Ensure the required libraries are installed using the provided requirements.txt file.
2. Run the Streamlit app using the following command:
```bash
streamlit run your_dashboard_script.py
```
Replace your_dashboard_script.py with the name of your Python script containing the Streamlit dashboard code.

## Additional Information
- The sentiment analysis script cleans and processes text data, providing sentiment scores and labels.
- The Streamlit dashboard visualizes sentiment patterns with various plots and charts.
- Contributions and feedback are welcome. Feel free to open issues or submit pull requests.