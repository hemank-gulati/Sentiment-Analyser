import pandas as pd
import numpy as np
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import streamlit as st
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')
stopwords = nltk.corpus.stopwords.words('english')
sid = SentimentIntensityAnalyzer()
st.title('Sentiment Analysis App')
st.subheader('By: Mridul Gulati')
st.write('This is a simple example of a Streamlit app that performs sentiment analysis on a given text or CSV file.')

with st.expander('Analyse Text'):
    text = st.text_area('Enter some text')
    if text and st.button('Analyse'):
        tokens = nltk.word_tokenize(text)
        filtered_tokens = [token for token in tokens if token not in stopwords]
        normalized_text = ' '.join(filtered_tokens)
        score = sid.polarity_scores(normalized_text)

        if score['compound'] >= 0.05:
            st.success("The text has a Positive sentiment.")
        elif score['compound'] <= -0.05:
            st.error("The text has a Negative sentiment.")
        else:
            st.warning("The text has a Neutral sentiment.")

with st.expander('Analyse CSV'):
    st.warning('The CSV file should contain a column with the name "Text"')
    uploaded_file = st.file_uploader('Choose a CSV file', type='csv')
    if uploaded_file and st.button('Analyse'):
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            df.dropna(subset=['Text'], inplace=True)
            df['score'] = df['Text'].apply(lambda x: sid.polarity_scores(x)['compound'])
            df['sentiment'] = np.where(df['score'] >= 0.05, 'Positive', np.where(df['score'] <= -0.05, 'Negative', 'Neutral') )
            st.write(df.head(3))
            @st.cache
            def get_sentiment_distribution(df):
                return df.to_csv("analysed.csv",index=False)
            st.write('Download the analysed file')
            st.download_button('Download', data=df.to_csv(index=False), file_name='analysed.csv', mime='text/csv')
        else:
            st.error('Please upload a CSV file')
    