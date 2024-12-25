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


def sentiment(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    import tensorflow as tf
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
    df = pd.read_csv('dataset.csv')
    import nltk
    from nltk.corpus import stopwords
##nltk.download('stopwords')
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
    df['Text'] = df['Text'].astype(str)
    df['Text'] = df['Text'].apply(clean_text)
    from nltk.stem import WordNetLemmatizer
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    lemmatizer = WordNetLemmatizer()
    def lemmatize_text(text):
        words = text.split()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
        return " ".join(lemmatized_words)

    df['Text']=df['Text'].apply(lemmatize_text)
    X = df['Text']  
    y = df['Sentiment'] 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    vocab_size = 5000  
    max_length = 200  
    embedding_dim = 100
    tokenizer = Tokenizer(num_words=vocab_size, oov_token='')
    tokenizer.fit_on_texts(X_train)
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    X_train_padded = pad_sequences(X_train_seq, maxlen=max_length, padding='post', truncating='post')
    X_test_padded = pad_sequences(X_test_seq, maxlen=max_length, padding='post', truncating='post')
    model = keras.Sequential([
    keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    keras.layers.Bidirectional(keras.layers.LSTM(64)),
    keras.layers.Dense(24, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    history = model.fit(
        X_train_padded,  
        y_train,       
        epochs=5,      
        verbose=1,
        validation_split=0.1
    )
    def predict_(text):
   
        cleaned_text = clean_text(text)  
    
    # Tokenize the text
        tokenized_text = tokenizer.texts_to_sequences([cleaned_text])
    
   
        padded_text = pad_sequences(tokenized_text, maxlen=max_length, padding='post', truncating='post')
        prediction = model.predict(padded_text)
        if 0.45 < prediction[0] < 0.55:
            return "Neutral"

        sentiment = "Positive" if prediction[0] > 0.5 else "Negative"

        return f"{sentiment}"
    def analyse(df, user):
        positive = 0
        negative = 0
        neutral=0
    # Filter rows based on the user variable (use user instead of the string 'user')
        res = df[df['user'] == user]
        res['message'] = res['message'].astype(str).fillna('')
    
    # Iterate through the filtered DataFrame
        for i in range(len(res)):
            text = res['message'].iloc[i]
            if predict_(text) == 'Positive':
                positive += 1
            elif predict_(text) == 'Negative':
                negative += 1
            else:
                neutral +=1

    # Calculate positive percentage
            positive_percentage = (positive + neutral) / (positive + negative+neutral) * 100 if (positive + negative) > 0 else 0
            negative_percentage = (negative) / (positive + negative+neutral) * 100 if (positive + negative) > 0 else 0
    
            return f"Positive  % of {user} is {positive_percentage:.2f}%     Negative % of {user} is {negative_percentage}%"

        

