
# coding: utf-8


import pandas as pd
import numpy as np

# for this section, input 'train.ft.txt' and 'test.ft.txt' seperately
# train.ft.txt and text.ft.txt are from Amazon Reviews
df = pd.read_csv('test.ft.txt', sep = '\n', header = None)
label_ls = []
for i in df[0][0:]:
    label = i[9:10]
    label_ls.append(label)  
len(label_ls)
df['label'] = pd.Series(label_ls).values
df[0].replace({'__label__2': ''}, inplace=True, regex=True)
df[0].replace({'__label__1': ''}, inplace=True, regex=True)
df.to_csv('amazon_test.csv', header = True, sep = ';', index=False)


# Trump's Tweets Dataset
df = pd.read_csv('trump.txt', sep = '\n', encoding = "ISO-8859-1")
print(df.shape)
ls = df['text,created_at'].tolist()

time_ls = []
content_ls = []
year_ls = []
date_ls = []
month_ls = []

for i in ls:
    time = i[-19:]
    year = i[-13:-9]
    date = i[-16:-14]
    month = i[-19:-17]
    content = i[:-20]
    
    date_ls.append(date)
    month_ls.append(month)
    time_ls.append(time)
    year_ls.append(year)
    content_ls.append(content)
print(len(time_ls))
df = pd.DataFrame()
df['content'] = pd.Series(content_ls).values
df['date'] = pd.Series(date_ls).values
df['year'] = pd.Series(year_ls).values
df['month'] = pd.Series(month_ls).values
df['time'] = pd.Series(time_ls).values
df.to_csv('trump.csv', header = True, sep = ';', index=False)
print(df.head())


# Trump's 200 tweets with labels
df = pd.read_csv('trump_a.txt', sep = '\n',  header = None, encoding = "ISO-8859-1")
ls = df[0].tolist()

label_ls = []
content_ls = []
for i in ls:
    label = i[0]
    label_ls.append(label)
    content = i[2:-30]
    content_ls.append(content)
    
df = pd.DataFrame()
df['content'] = pd.Series(content_ls).values
df['label'] = pd.Series(label_ls).values
df.to_csv('trump_label.csv', header = True, sep = ';', index=False)
print(df.head())

