import re
import pandas as pd
def preprocess(data):
    pattern = '\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s?[APMapm]{2}'
    messages = re.split(pattern,data)[1:]
    dates = re.findall(pattern,data)
    df= pd.DataFrame({'user-message':messages,'message_date':dates})
    ##convert message to date type format
    df['message_date']= pd.to_datetime(df['message_date'],format='%d/%m/%Y, %I:%M %p')

    df.rename(columns={'message_date':'date'},inplace = True)
    users = []
    messages = []

    for message in df['user-message']:
       entry = re.split(r'([\w\s]+):\s', message)  # Adjusted regex to match the user and message format
       if len(entry) > 2:  # Check if there is a user name and a message
            users.append(entry[1])  # User name
            messages.append(entry[2])  # User message
       else:
            users.append('group_notification')  # Default user if no name found
            messages.append(entry[0])  # The entire message

    df['user'] = users
    df['message'] = messages
    df.drop(columns=['user-message'], inplace=True)
    
    
    df['only_date'] = df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute


    return df