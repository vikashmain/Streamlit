import re
import pandas as pd

def preprocess(data):
    # Regular expression pattern to extract dates
    pattern = r'\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s?[APMapm]{2}'
    
    # Extract messages and dates
    messages = re.split(pattern, data)[1:]  # Skip the first split part as it's empty
    dates = re.findall(pattern, data)
    
    # Create DataFrame
    df = pd.DataFrame({'user-message': messages, 'message_date': dates})
    
    # Replace non-breaking spaces and standardize format
    df['message_date'] = df['message_date'].str.replace('\u202f', ' ')
    
    # Convert 'message_date' to datetime
    try:
        df['message_date'] = pd.to_datetime(df['message_date'], format='%d/%m/%y, %I:%M %p', dayfirst=True, errors='coerce')
    except ValueError as e:
        raise ValueError(f"Error in date parsing: {e}")
    
    # Rename 'message_date' column to 'date'
    df.rename(columns={'message_date': 'date'}, inplace=True)
    
    # Extract users and messages
    users = []
    messages = []
    
    for message in df['user-message']:
        # Regex to split user and message; captures "<user>: <message>"
        entry = re.split(r'([\w\s]+):\s', message)
        if len(entry) > 2:  # Check if there is a user name and a message
            users.append(entry[1].strip())  # User name
            messages.append(entry[2].strip())  # User message
        else:
            users.append('group_notification')  # Default for system messages
            messages.append(entry[0].strip())  # Entire message
    
    # Assign to DataFrame
    df['user'] = users
    df['message'] = messages
    
    # Drop the original 'user-message' column
    df.drop(columns=['user-message'], inplace=True)
    
    # Add additional date-related columns
    df['only_date'] = df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    
    # Return the processed DataFrame
    return df
