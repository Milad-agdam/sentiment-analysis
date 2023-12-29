import streamlit as st
import pandas as pd

# Load your data into a Pandas DataFrame
df = pd.read_csv('data.csv')


def get_top_n_gram(tweet_df, ngram_range, n=10):
    # load the stopwords
    stopwords = set()
    with open("static/en_stopwords_viz.txt", "r") as file:
        for word in file:
            stopwords.add(word.rstrip("\n"))

    # load the corpus and vectorizer
    corpus = tweet_df["Cleaned Tweet"]
    vectorizer = CountVectorizer(
        analyzer="word", ngram_range=ngram_range, stop_words=stopwords
    )

    # use the vectorizer to count the n-grams frequencies
    X = vectorizer.fit_transform(corpus.astype(str).values)
    words = vectorizer.get_feature_names_out()
    words_count = np.ravel(X.sum(axis=0))

    # store the results in a dataframe
    df = pd.DataFrame(zip(words, words_count))
    df.columns = ["words", "counts"]
    df = df.sort_values(by="counts", ascending=False).head(n)
    df["words"] = df["words"].str.title()
    return df

def plot_n_gram(n_gram_df, title, color="#54A24B"):
    # plot the top n-grams frequencies in a bar chart
    fig = px.bar(
        x=n_gram_df.counts,
        y=n_gram_df.words,
        title="<b>{}</b>".format(title),
        text_auto=True,
    )
    fig.update_layout(plot_bgcolor="white")
    fig.update_xaxes(title=None)
    fig.update_yaxes(autorange="reversed", title=None)
    fig.update_traces(hovertemplate="<b>%{y}</b><br>Count=%{x}", marker_color=color)
    return fig






st.set_page_config(
    page_title="Sentiment Analyzer", page_icon="📊", layout="wide"
)

adjust_top_pad = """
    <style>
        div.block-container {padding-top:1rem;}
    </style>
    """
st.markdown(adjust_top_pad, unsafe_allow_html=True)


with st.sidebar:
    st.title("Sentiment Analyzer")

    st.markdown(
        """
        <div style="text-align: justify;">
            Explore and analyze the sentiment of user comments with
            this interactive dashboard. Gain insights into the distribution of positive, negative,
            and neutral sentiments in your dataset. Fine-tune sentiment labels, visualize
            percentage breakdowns and delve into the raw data
            for a comprehensive understanding of the sentiments expressed.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # create a form to obtain the search parameters
    with st.form(key="search_form"):
        st.subheader("Search Parameters")
        # session_state.search_term will be updated when the form is submitted
        st.text_input("Search term", key="search_term")
        # session_state.num_tweets will be updated when the form is submitted
        st.slider("Number of tweets", min_value=100, max_value=2000, key="num_tweets")
        # search_callback will be called when the form is submitted
        st.form_submit_button(label="Search")
        st.markdown(
            "Note: it may take a while to load the results, especially with large number of tweets"
        )

    st.markdown("Created by Milad Agdam")




# create 3 tabs for all, positive, and negative tweets
tab1, tab2, tab3 = st.tabs(["All", "Positive 😊", "Negative ☹️"])
with tab1:
    # make dashboard for all tweets
    tweet_df = st.session_state.df
    make_dashboard(tweet_df, bar_color="#54A24B", wc_color="Greens")

with tab2:
    # make dashboard for tweets with positive sentiment
    tweet_df = st.session_state.df.query("Sentiment == 'Positive'")
    make_dashboard(tweet_df, bar_color="#1F77B4", wc_color="Blues")

with tab3:
    # make dashboard for tweets with negative sentiment
    tweet_df = st.session_state.df.query("Sentiment == 'Negative'")
