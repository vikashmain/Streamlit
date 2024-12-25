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
##nltk.download('stopwords')

df = pd.read_csv('/Users/Vikash/Desktop/ChatAnalysis/Sentiment/dataset.csv')

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
##nltk.download('wordnet')
##nltk.download('omw-1.4')

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
df['Text'] = df['Text'].apply(lambda x: lemmatize_text(str(x)))  # Convert all to string first

X = df['Text']  
y = df['Sentiment'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


vocab_size = 5000  
max_length = 200  
embedding_dim = 100
tokenizer = Tokenizer(num_words=vocab_size, oov_token='')
tokenizer.fit_on_texts(X_train)

# X_train_seq = tokenizer.texts_to_sequences(X_train)
# X_test_seq = tokenizer.texts_to_sequences(X_test)

# X_train_padded = pad_sequences(X_train_seq, maxlen=max_length, padding='post', truncating='post')
# X_test_padded = pad_sequences(X_test_seq, maxlen=max_length, padding='post', truncating='post')

# model = keras.Sequential([
#     keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
#     keras.layers.Bidirectional(keras.layers.LSTM(64)),
#     keras.layers.Dense(24, activation='relu'),
#     keras.layers.Dense(1, activation='sigmoid')
# ])

# model.compile(loss='binary_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])
# history = model.fit(
#     X_train_padded,  
#     y_train,       
#     epochs=5,      
#     verbose=1,
#     validation_split=0.1
# )

# model.save('model.keras')

from tensorflow.keras.models import load_model

model = load_model('model.keras')


def predict_(text):
    text = str(text)
    cleaned_text = clean_text(text)  
    
    # Tokenize the text
    tokenized_text = tokenizer.texts_to_sequences([cleaned_text])
    
   
    padded_text = pad_sequences(tokenized_text, maxlen=200, padding='post', truncating='post')
    prediction = model.predict(padded_text)
    if 0.48 < prediction[0] < 0.55:
        return f"Neutral"

    sentiment = "Negative" if prediction[0] < 0.40 else "Positive"

    return f"{sentiment}"

data = pd.read_csv('/Users/Vikash/Desktop/ChatAnalysis/Sentiment/data.csv')

import pandas as pd

def analyse(data, user):
    user_messages = data[data['user'] == user]['message']
    
    message_list = user_messages.tolist()
    
    positive_count = 0
    negative_count = 0
    neutral_count = 0
    
    for message in message_list:
        sentiment = predict_(message) 
        if sentiment == 'Positive':
            positive_count += 1
        elif sentiment == 'Negative':
            negative_count += 1
        else:
            neutral_count += 1
    
    
    total_messages = len(message_list)
    
   
    a=  [{
        'user': user,
        'total_messages': total_messages,
        'positive': positive_count,
        'negative': negative_count,
        'neutral': neutral_count,
    }]
    b = pd.DataFrame(a)
    return b
print(analyse(data,' Saurabh Singh'))