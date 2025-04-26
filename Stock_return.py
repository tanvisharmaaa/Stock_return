#!/usr/bin/env python
# coding: utf-8

# # Getting the Data

# In[1]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 


# In[2]:


# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 16:02:02 2021

@author: epinsky
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 14:37:29 2018

@author: epinsky
"""

# install yfinance version 0.1.62
#   !pip install yfinance==0.1.62
#pip install pandas_datareader
from pandas_datareader import data as web
import os
import pandas as pd
import yfinance as yf

def get_stock(ticker, start_date, end_date, s_window, l_window):
    try:
#       yf.pdr_override()
        df = yf.download(ticker, start=start_date, end=end_date)
# can use this as well        df = web.get_data_yahoo(ticker, start=start_date, end=end_date)
        df['Return'] = df['Adj Close'].pct_change()
        df['Return'].fillna(0, inplace = True)
        df['Date'] = df.index
        df['Date'] = pd.to_datetime(df['Date'])
        df['Month'] = df['Date'].dt.month
        df['Year'] = df['Date'].dt.year 
        df['Day'] = df['Date'].dt.day
        for col in ['Open', 'High', 'Low', 'Close', 'Adj Close']:
            df[col] = df[col].round(2)
        df['Weekday'] = df['Date'].dt.day_name()
        df['Week_Number'] = df['Date'].dt.strftime('%U')
        df['Year_Week'] = df['Date'].dt.strftime('%Y-%U')
        df['Short_MA'] = df['Adj Close'].rolling(window=s_window, min_periods=1).mean()
        df['Long_MA'] = df['Adj Close'].rolling(window=l_window, min_periods=1).mean()        
        col_list = ['Date', 'Year', 'Month', 'Day', 'Weekday', 
                    'Week_Number', 'Year_Week', 'Open', 
                    'High', 'Low', 'Close', 'Volume', 'Adj Close',
                    'Return', 'Short_MA', 'Long_MA']
        num_lines = len(df)
        df = df[col_list]
        print('read ', num_lines, ' lines of data for ticker: ' , ticker)
        return df
    except Exception as error:
        print(error)
        return None

try:
    
    ticker='TSLA' # SPY as well 
    input_dir = os.getcwd()
    output_file = os.path.join(input_dir, ticker + '.csv')
    df = get_stock(ticker, start_date='2018-01-01', end_date='2022-12-31', 
               s_window=14, l_window=50)
    df.to_csv(output_file, index=False)
    print('wrote ' + str(len(df)) + ' lines to file: ' + output_file)
except Exception as e:
    print(e)
    print('failed to get Yahoo stock data for ticker: ', ticker)


# # Question 1

# In[2]:


df = pd.read_csv('TSLA.csv')


# In[3]:


df_spy = pd.read_csv('SPY.csv')


# In[4]:


#df_s


# In[4]:


df_short = df[['Date','Year', 'Return']]
df_short_spy = df_spy[['Date', 'Year', 'Return']]
#df_short


# In[5]:


df_short['Date'] = pd.to_datetime(df['Date'])
df_short_spy['Date'] = pd.to_datetime(df['Date'])


# In[6]:


train_df = df_short[(df_short['Year'] >= 2018) & (df_short['Year'] <= 2020)]
test_df = df_short[(df_short['Year'] >= 2021) & (df_short['Year'] <= 2022)]

#Spy Details 

train_df_spy = df_short_spy[(df_short_spy['Year'] >= 2018) & (df_short_spy['Year'] <= 2020)]
test_df_spy = df_short_spy[(df_short_spy['Year'] >= 2021) & (df_short_spy['Year'] <= 2022)]


# In[7]:


#train_df
#test_df

#train_df_spy
#test_df_spy


# In[8]:


train_df['Sign'] = train_df['Return'].apply(lambda x: '+' if x >= 0 else '-')
test_df['Sign'] = test_df['Return'].apply(lambda x: '+' if x >= 0 else '-')

train_df_spy['Sign'] = train_df_spy['Return'].apply(lambda x: '+' if x >= 0 else '-')
test_df_spy['Sign'] = test_df_spy['Return'].apply(lambda x: '+' if x >= 0 else '-')


# In[10]:


#train_df


# In[11]:


l_minus= train_df[train_df['Sign'] == '-']['Sign'].count()
l_plus = train_df[train_df['Sign'] == '+']['Sign'].count()

l_minus_spy= train_df_spy[train_df_spy['Sign'] == '-']['Sign'].count()
l_plus_spy = train_df_spy[train_df_spy['Sign'] == '+']['Sign'].count()


# In[12]:


#l_minus
#l_plus

#l_minus_spy
#l_plus_spy


# In[13]:


p_star = l_plus/len(train_df)

p_star_spy = l_plus_spy/len(train_df_spy)


# In[14]:


print('Probability for Tesla: %f' %p_star)
print('Probability for SPY: %f' %p_star_spy)


# In[15]:


## 


# In[15]:


def calculate_probability(k):
    consecutive_down_plus = 0
    consecutive_down = 0

    for i in range(len(train_df) - k):
        pattern = ''.join(['-' for _ in range(k)]) + '+'
        if train_df['Sign'].iloc[i:i+k+1].str.cat() == pattern:
            consecutive_down_plus += 1
        if train_df['Sign'].iloc[i:i+k].str.cat() == '-' * k:
            consecutive_down += 1

    probability = consecutive_down_plus / consecutive_down if consecutive_down > 0 else 0
    return probability


k_values = [1, 2, 3]
for k in k_values:
    probability = calculate_probability(k)
    print(f"Probability that after seeing {k} consecutive '-', the next day is a '+': {probability:.2f}")
    
    
    


# In[16]:


##SPy 

def calculate_probability_spy(k):
    consecutive_down_plus_spy = 0
    consecutive_down_spy = 0

    for i in range(len(train_df_spy) - k):
        pattern = ''.join(['-' for _ in range(k)]) + '+'
        if train_df_spy['Sign'].iloc[i:i+k+1].str.cat() == pattern:
            consecutive_down_plus_spy += 1
        if train_df_spy['Sign'].iloc[i:i+k].str.cat() == '-' * k:
            consecutive_down_spy += 1

    probability_spy = consecutive_down_plus_spy / consecutive_down_spy if consecutive_down_spy > 0 else 0
    return probability_spy

# Calculate the probabilities for k = 1, 2, and 3
k_values = [1, 2, 3]
for k in k_values:
    probability_spy = calculate_probability_spy(k)
    print(f"Probability that after seeing {k} consecutive '-', the next day is a '+': {probability_spy:.2f}")


# In[17]:


#tesla

def calculate_probability(k):
    consecutive_up_plus = 0
    consecutive_up = 0

    for i in range(len(df) - k):
        pattern = '+' * (k + 1)
        if train_df['Sign'].iloc[i:i+k+1].str.cat() == pattern:
            consecutive_up_plus += 1
        if train_df['Sign'].iloc[i:i+k].str.cat() == '+' * k:
            consecutive_up += 1

    probability = consecutive_up_plus / consecutive_up if consecutive_up > 0 else 0
    return probability

# Calculate the probabilities for k = 1, 2, and 3
k_values = [1, 2, 3]
for k in k_values:
    probability = calculate_probability(k)
    print(f"Probability that after seeing {k} consecutive '+', the next day is still a '+': {probability:.2f}")


# In[18]:


#Spy

def calculate_probability_spy(k):
    consecutive_up_plus_spy = 0
    consecutive_up_spy = 0

    for i in range(len(df) - k):
        pattern = '+' * (k + 1)
        if train_df_spy['Sign'].iloc[i:i+k+1].str.cat() == pattern:
            consecutive_up_plus_spy += 1
        if train_df_spy['Sign'].iloc[i:i+k].str.cat() == '+' * k:
            consecutive_up_spy += 1

    probability = consecutive_up_plus_spy / consecutive_up_spy if consecutive_up_spy > 0 else 0
    return probability

# Calculate the probabilities for k = 1, 2, and 3
k_values = [1, 2, 3]
for k in k_values:
    probability = calculate_probability_spy(k)
    print(f"Probability that after seeing {k} consecutive '+', the next day is still a '+': {probability:.2f}")


# # Predicting labels

# In[19]:


true_sign_train = train_df['Sign']
true_sign_test = test_df['Sign']
predicted_labels_dict = {}

def predict_labels(true_labels_train, true_labels_test, W):
    predicted_labels = []

    for i in range(W, len(true_labels_test)):
       
        sequence = tuple(true_labels_test[i - W:i])

       
        count_minus = sum(1 for j in range(len(true_labels_train) - W) if tuple(true_labels_train[j:j+W]) == sequence and true_labels_train[j + W] == '-')
        count_plus = sum(1 for j in range(len(true_labels_train) - W) if tuple(true_labels_train[j:j+W]) == sequence and true_labels_train[j + W] == '+')

     
        if count_plus >= count_minus:
            predicted_labels.append('+')
        else:
            predicted_labels.append('-')

    return predicted_labels


W_values = [2, 3, 4]


for W in W_values:
    predicted_labels_dict[f'Predicted_Labels_W{W}'] = predict_labels(true_sign_train, true_sign_test, W)
    


# In[20]:


## Spy 
true_sign_train_spy = train_df_spy['Sign']
true_sign_test_spy = test_df_spy['Sign']
predicted_labels_dict_spy = {}


def predict_labels_spy(true_labels_train_spy, true_labels_test_spy, W):
    predicted_labels_spy = []

    for i in range(W, len(true_labels_test_spy)):

        sequence = tuple(true_labels_test_spy[i - W:i])

       
        count_minus_spy = sum(1 for j in range(len(true_labels_train_spy) - W) if tuple(true_labels_train_spy[j:j+W]) == sequence and true_labels_train_spy[j + W] == '-')
        count_plus_spy = sum(1 for j in range(len(true_labels_train_spy) - W) if tuple(true_labels_train_spy[j:j+W]) == sequence and true_labels_train_spy[j + W] == '+')

    
        if count_plus_spy >= count_minus_spy:
            predicted_labels_spy.append('+')
        else:
            predicted_labels_spy.append('-')

    return predicted_labels_spy


W_values = [2, 3, 4]

for W in W_values:
    predicted_labels_dict_spy[f'Predicted_Labels_W{W}'] = predict_labels_spy(true_sign_train_spy, true_sign_test_spy, W)


# In[21]:


#TESLA

predicted_df = pd.DataFrame()

# Loop through the W values and the corresponding predicted labels
for W in W_values:
    # Get the predicted labels for the current W
    predicted_labels = predicted_labels_dict[f'Predicted_Labels_W{W}']

    # Pad the initial values with NaN
    num_nan = W # Number of NaN values to add
    padded_labels = [np.nan] * num_nan + predicted_labels

    # Add the padded labels to the DataFrame
    predicted_df[f'Predicted_Labels_W{W}'] = padded_labels

# Print the resulting DataFrame
#print(predicted_df)


# In[22]:


# SPY

predicted_df_spy = pd.DataFrame()

# Loop through the W values and the corresponding predicted labels
for W in W_values:
    # Get the predicted labels for the current W
    predicted_labels_spy = predicted_labels_dict_spy[f'Predicted_Labels_W{W}']

    # Pad the initial values with NaN
    num_nan_spy = W # Number of NaN values to add
    padded_labels_spy = [np.nan] * num_nan_spy + predicted_labels_spy
    
    # Add the padded labels to the DataFrame
    predicted_df_spy[f'Predicted_Labels_W{W}'] = padded_labels_spy

# Print the resulting DataFrame
#print(predicted_df_spy)


# In[23]:


test_df = test_df.reset_index(drop=True)
predicted_df = predicted_df.reset_index(drop=True)

# Concatenate test_df and predicted_df along the columns axis
merged_df = pd.concat([test_df, predicted_df], axis=1)

# Print the resulting merged DataFrame
#print(merged_df)


# In[24]:


test_df_spy = test_df_spy.reset_index(drop=True)
predicted_df_spy = predicted_df_spy.reset_index(drop=True)

# Concatenate test_df and predicted_df along the columns axis
merged_df_spy = pd.concat([test_df_spy, predicted_df_spy], axis=1)

# Print the resulting merged DataFrame
#print(merged_df)


# In[26]:


merged_df


# # Accuracy for each sign 

# In[27]:


W_values = ['W2', 'W3', 'W4']

# Create a DataFrame to store accuracy for each W value for '+' and '-' separately
accuracy_df = pd.DataFrame({'W': W_values, '+ Accuracy': [0] * len(W_values), '- Accuracy': [0] * len(W_values)})


def calculate_accuracy(w_value, sign):
    ensemble_label_col = 'Predicted_Labels_' + w_value
    mask = (merged_df['Sign'] == sign) & (merged_df[ensemble_label_col] == sign)
    accuracy = (mask.sum() / (merged_df['Sign'] == sign).sum()) * 100
    return accuracy


accuracy_df['+ Accuracy TSLA'] = accuracy_df['W'].apply(lambda w: calculate_accuracy(w, '+'))
accuracy_df['- Accuracy TSLA'] = accuracy_df['W'].apply(lambda w: calculate_accuracy(w, '-'))


print(accuracy_df)


# In[28]:


W_values = ['W2', 'W3', 'W4']

# Create a DataFrame to store accuracy for each W value for '+' and '-' separately
accuracy_df_spy = pd.DataFrame({'W': W_values, '+ Accuracy': [0] * len(W_values), '- Accuracy': [0] * len(W_values)})

# Function to calculate accuracy for a specific W value and sign
def calculate_accuracy_spy(w_value, sign):
    ensemble_label_col = 'Predicted_Labels_' + w_value
    mask = (merged_df_spy['Sign'] == sign) & (merged_df_spy[ensemble_label_col] == sign)
    accuracy_spy = (mask.sum() / (merged_df_spy['Sign'] == sign).sum()) * 100
    return accuracy_spy

# Calculate accuracy for each W value for '+' and '-' separately
accuracy_df_spy['+ Accuracy SPY'] = accuracy_df_spy['W'].apply(lambda w: calculate_accuracy_spy(w, '+'))
accuracy_df_spy['- Accuracy SPY'] = accuracy_df_spy['W'].apply(lambda w: calculate_accuracy_spy(w, '-'))

# Display the accuracy_df
print(accuracy_df_spy)


# # Question 3 / Ensemble Learning

# In[32]:


def compute_ensemble_label(row):
    labels = row[['Predicted_Labels_W2', 'Predicted_Labels_W3', 'Predicted_Labels_W4']]
    label_counts = labels.value_counts()
    if label_counts.empty:
        return np.nan  # Handle the case where all predictions are NaN
    return label_counts.idxmax()

# Apply the compute_ensemble_label function to each row to calculate the ensemble label
merged_df['Ensemble_Label'] = merged_df.apply(compute_ensemble_label, axis=1)

# Print the accuracy_df with the new 'Ensemble_Label' column
#print(merged_df)


# In[33]:


## SPY

def compute_ensemble_label_spy(row):
    labels = row[['Predicted_Labels_W2', 'Predicted_Labels_W3', 'Predicted_Labels_W4']]
    label_counts = labels.value_counts()
    if label_counts.empty:
        return np.nan  # Handle the case where all predictions are NaN
    return label_counts.idxmax()

# Apply the compute_ensemble_label function to each row to calculate the ensemble label
merged_df_spy['Ensemble_Label'] = merged_df_spy.apply(compute_ensemble_label, axis=1)

# Print the accuracy_df with the new 'Ensemble_Label' column
print(merged_df_spy)


# In[34]:


#merged_df
#merged_df_spy


# In[39]:


accuracy_plus = (merged_df[merged_df['Sign'] == '+']['Sign'] == merged_df[merged_df['Sign'] == '+']['Ensemble_Label']).mean() * 100
accuracy_minus = (merged_df[merged_df['Sign'] == '-']['Sign'] == merged_df[merged_df['Sign'] == '-']['Ensemble_Label']).mean() * 100

print(f"Accuracy for '+': {accuracy_plus:.2f}%")
print(f"Accuracy for '-': {accuracy_minus:.2f}%")


# In[40]:


accuracy_plus_spy = (merged_df_spy[merged_df['Sign'] == '+']['Sign'] == merged_df_spy[merged_df['Sign'] == '+']['Ensemble_Label']).mean() * 100
accuracy_minus_spy = (merged_df_spy[merged_df['Sign'] == '-']['Sign'] == merged_df_spy[merged_df['Sign'] == '-']['Ensemble_Label']).mean() * 100

print(f"Accuracy for '+': {accuracy_plus_spy:.2f}%")
print(f"Accuracy for '-': {accuracy_minus_spy:.2f}%")


# # Question 4

# In[41]:


def compute_statistics(data, label_col, true_col):
    TP = np.sum((data[label_col] == '+') & (data[true_col] == '+'))
    FP = np.sum((data[label_col] == '+') & (data[true_col] == '-'))
    TN = np.sum((data[label_col] == '-') & (data[true_col] == '-'))
    FN = np.sum((data[label_col] == '-') & (data[true_col] == '+'))
    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)
    return TP, FP, TN, FN, TPR, TNR

# Calculate statistics for each value of W and the ensemble label
W_values = [2, 3, 4, 'Ensemble_Label']
for w in W_values:
    label_col = f'Predicted_Labels_W{w}' if w != 'Ensemble_Label' else 'Ensemble_Label'
    TP, FP, TN, FN, TPR, TNR = compute_statistics(merged_df, label_col, 'Sign')
    print(f'Statistics for W={w}:')
    print(f'TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}')
    print(f'TPR (True Positive Rate): {TPR:.2f}')
    print(f'TNR (True Negative Rate): {TNR:.2f}')
    print()


# In[42]:


def compute_statistics_spy(data, label_col, true_col):
    TP_spy = np.sum((data[label_col] == '+') & (data[true_col] == '+'))
    FP_spy = np.sum((data[label_col] == '+') & (data[true_col] == '-'))
    TN_spy = np.sum((data[label_col] == '-') & (data[true_col] == '-'))
    FN_spy = np.sum((data[label_col] == '-') & (data[true_col] == '+'))
    TPR_spy = TP_spy / (TP_spy + FN_spy)
    TNR_spy = TN_spy / (TN_spy + FP_spy)
    return TP_spy, FP_spy, TN_spy, FN_spy, TPR_spy, TNR_spy

# Calculate statistics for each value of W and the ensemble label
W_values = [2, 3, 4, 'Ensemble_Label']
for w in W_values:
    label_col_spy = f'Predicted_Labels_W{w}' if w != 'Ensemble_Label' else 'Ensemble_Label'
    TP_spy, FP_spy, TN_spy, FN_spy, TPR_spy, TNR_spy = compute_statistics(merged_df_spy, label_col_spy, 'Sign')
    print(f'Statistics for W={w}:')
    print(f'TP: {TP_spy}, FP: {FP_spy}, TN: {TN_spy}, FN: {FN_spy}')
    print(f'TPR (True Positive Rate): {TPR_spy:.2f}')
    print(f'TNR (True Negative Rate): {TNR_spy:.2f}')
    print()


# In[61]:


def map_labels(label):
    if label == '+':
        return 1
    elif label == '-':
        return -1
    else:
        return 0

merged_df['Predicted_Labels_W2'] = merged_df['Predicted_Labels_W2'].apply(map_labels)
merged_df['Ensemble_Label'] = merged_df['Ensemble_Label'].apply(map_labels)

# Initialize variables
initial_balance = 100
balance_w2 = [initial_balance]
balance_ensemble = [initial_balance]
balance_buy_and_hold = [initial_balance]

for index, row in merged_df.iterrows():
    # Calculate balances based on trading decisions and returns
    balance_w2.append(balance_w2[-1] * (1 + row['Return'] * row['Predicted_Labels_W2']))
    balance_ensemble.append(balance_ensemble[-1] * (1 + row['Return'] * row['Ensemble_Label']))
    
    # Calculate buy-and-hold portfolio value
    balance_buy_and_hold.append(balance_buy_and_hold[-1] * (1 + row['Return']))

date_index = pd.to_datetime(merged_df['Date'])

plt.figure(figsize=(12, 6))
plt.plot(date_index, balance_w2[:-1], label='Trading with W2', linestyle='--')
plt.plot(date_index, balance_ensemble[:-1], label='Trading with Ensemble', linestyle='-')
plt.plot(date_index, balance_buy_and_hold[:-1], label='Buy-and-Hold', linestyle='-')

plt.title('Portfolio Growth Over Time')
plt.xlabel('Date')
plt.ylabel('Portfolio Value ($)')
plt.legend()
plt.grid(True)
plt.show()


# In[43]:


merged_df


# In[44]:


merged_df = merged_df.fillna(0)

# Define a function to map '+' to +1 and '-' to -1
def map_labels(label):
    if label == '+':
        return 1
    elif label == '-':
        return -1
    else:
        return 0


merged_df['Predicted_Labels_W2'] = merged_df['Predicted_Labels_W2'].apply(map_labels)

merged_df['Ensemble_Label'] = merged_df['Ensemble_Label'].apply(map_labels)

# Initialize variables
initial_balance = 100  
balance_w2 = [initial_balance]  

balance_ensemble = [initial_balance]  

for index, row in merged_df.iterrows():
    balance_w2.append(balance_w2[-1] * (1 + row['Return'] * row['Predicted_Labels_W2']))

    balance_ensemble.append(balance_ensemble[-1] * (1 + row['Return'] * row['Ensemble_Label']))


date_index = pd.to_datetime(merged_df['Date'])


plt.figure(figsize=(12, 6))
plt.plot(date_index, balance_w2[:-1], label='Trading with W2', linestyle='--')

plt.plot(date_index, balance_ensemble[:-1], label='Trading with Ensemble', linestyle='-')
plt.title('Portfolio Growth Over Time')
plt.xlabel('Date')
plt.ylabel('Portfolio Value ($)')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




