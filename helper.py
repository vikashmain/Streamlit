from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import emoji

extract = URLExtract()

def fetch_stats(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # fetch the number of messages
    num_messages = df.shape[0]

    # fetch the total number of words
    words = []
    for message in df['message']:
        words.extend(message.split())

    # fetch number of media messages
    num_media_messages = df[df['message'] == '<Media omitted>\n'].shape[0]

    # fetch number of links shared
    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))

    return num_messages,len(words),num_media_messages,len(links)

def most_busy_users(df):
    x = df['user'].value_counts().head()
    ## % of messages by users
    df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'name', 'user': 'percent'})
    return x,df

def create_wordcloud(selected_user,df):

    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)

    wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')
    temp['message'] = temp['message'].apply(remove_stop_words)
    df_wc = wc.generate(df['message'].str.cat(sep=" "))
    return df_wc

def most_common_words(selected_user,df):

    f = open('stop_hinglish.txt','r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    words = []
    ## optional: removing extra stopwords
    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df

def emoji_helper(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = []
    for message in df['message']:
        emojis.extend([c for c in message if c in emoji.EMOJI_DATA])


    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))

    return emoji_df

def monthly_timeline(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))

    timeline['time'] = time
   # gives count of mesages on a given month and year

    return timeline

def daily_timeline(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
   # gives count of mesages on a given date
    daily_timeline = df.groupby('only_date').count()['message'].reset_index()

    return daily_timeline

def week_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['day_name'].value_counts()

def month_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['month'].value_counts()


import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import math
import nltk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

##df = pd.read_csv('/Users/Vikash/Desktop/ChatAnalysis/Sentiment/dataset.csv')

stop_word = set(stopwords.words('english'))

import re
from bs4 import BeautifulSoup


def remove_html_tags(text):
    return BeautifulSoup(text, "html.parser").get_text()


def remove_urls(text):
    return re.sub(r'http\S+|www\S+', '', text)

def remove_non_alphanumeric(text):
    return re.sub(r'[^A-Za-z\s]', '', text)
    
def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_word]
    return  " ".join(filtered_words)
    
def clean_text(text):
    text = remove_html_tags(text)
    text = remove_urls(text)
    text = remove_non_alphanumeric(text)
    text = remove_stopwords(text)
    return text


from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()
import pandas as pd

# Example of the lemmatize_text function
def lemmatize_text(text):
    # Ensure the input is a string
    if isinstance(text, str):
        words = text.split()
        # Add your lemmatization code here (e.g., using NLTK or spaCy)
        # return lemmatized_text
        return ' '.join(words)  # Just an example, replace with actual lemmatization
    else:
        # If it's not a string (e.g., NaN or float), return an empty string or handle it as needed
        return ""

# Apply the lemmatize_text function to the 'Text' column of the dataframe
# df['Text'] = df['Text'].apply(lambda x: lemmatize_text(str(x)))  # Convert all to string first

# X = df['Text']  
# y = df['Sentiment'] 

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


vocab_size = 5000  
max_length = 200  
embedding_dim = 100
tokenizer = Tokenizer(num_words=vocab_size, oov_token='')
from tensorflow.keras.models import load_model

model = load_model('model.keras')
def clean_message(message):
    # Ensure the message is not None or empty
    if message is None or message == "":
        return ""
    return message

def predict_(text):
    # Ensure the text is a string
    text = str(text)
    
    # Clean the text
    cleaned_text = clean_text(text)  
    
    if not cleaned_text:  # Check if cleaned_text is empty or None
        print(f"Warning: Empty or invalid cleaned text for message: {text}")
        return "Neutral"  # Return a default value if the text is invalid
    
    # Tokenize the cleaned text
    tokenized_text = tokenizer.texts_to_sequences([cleaned_text])
    
    if not tokenized_text or not tokenized_text[0]:  # Check if tokenized text is empty
        print(f"Warning: Empty tokenized text for message: {text}")
        return "Neutral"  # Return a default value if the tokenized text is empty
    
    # Pad the tokenized text to the required length
    padded_text = pad_sequences(tokenized_text, maxlen=200, padding='post', truncating='post')
    
    # Predict sentiment using the model
    prediction = model.predict(padded_text)
    
    if prediction is None or len(prediction) == 0:  # Check if the model prediction is valid
        print(f"Warning: Invalid prediction for message: {text}")
        return "Neutral"  # Return a default value if the prediction is invalid
    
    # Inspect the prediction value to ensure it's a valid number
    print(f"Prediction value for message: {text} = {prediction[0]}")
    
    # Handle different prediction ranges
    if prediction[0] < 0.40:
        return "Negative"
    elif 0.48 < prediction[0] < 0.55:
        return "Neutral"
    else:
        return "Positive"

import pandas as pd

def analyse(data, user):
    # Filter messages for the selected user
    user_messages = data[data['user'] == user]['message']
    
    # Convert the messages to a list
    message_list = user_messages.tolist()
    
    # Initialize sentiment counters
    positive_count = 0
    negative_count = 0
    neutral_count = 0
    
    # Check if the user has any messages
    if not message_list:
        print(f"Warning: No messages found for user {user}")
        return pd.DataFrame([{
            'user': user,
            'total_messages': 0,
            'positive': 0,
            'negative': 0,
            'neutral': 0,
        }])

    # Iterate through each message and predict its sentiment
    for message in message_list:
        sentiment = predict_(message)
        if sentiment == 'Positive':
            positive_count += 1
        elif sentiment == 'Negative':
            negative_count += 1
        else:
            neutral_count += 1
    
    # Calculate the total number of messages
    total_messages = len(message_list)
    
    # Create a DataFrame with the results
    results = [{
        'user': user,
        'total_messages': total_messages,
        'positive': positive_count,
        'negative': negative_count,
        'neutral': neutral_count,
    }]
    
    return pd.DataFrame(results)
