
# coding: utf-8

import pandas as pd
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
import random
from gensim.models import FastText
import time
from keras.models import Sequential
from keras.layers import Conv1D, Dropout, Dense, Flatten, LSTM, MaxPooling1D, Bidirectional
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, TensorBoard

import keras.backend as K
import nltk
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#load data
df2 = pd.read_csv('amazon.csv', sep = ';')
df_pos = df2.loc[df2['label'] == 2]
df_neg = df2.loc[df2['label'] == 1]
print(df_pos.shape)
print(df_neg.shape)


# Visualize Positive & Negative word count

pos_ls = df_pos['0'].tolist()
neg_ls = df_neg['0'].tolist()

# average length of positive and negative reviews
print(sum(map(len, pos_ls))/len(pos_ls))
print(sum(map(len, neg_ls))/len(neg_ls))

each_len_pos = []
for i in pos_ls:
    each_len_pos.append(len(i))
    
each_len_neg = []
for i in neg_ls:
    each_len_neg.append(len(i))
  
# positive reviews
plt.hist(each_len_pos, bins=30)
plt.title('Word Count Distribution of Positive Amazon Reviews')
plt.ylabel('Frequency');

# negative reviews
plt.hist(each_len_neg, bins=30)
plt.title('Word Count Distribution of Negative Amazon Reviews')
plt.ylabel('Frequency');


# # FastText 

# Tokenizer: remove punctuations
tokenizer = RegexpTokenizer('[a-zA-Z0-9]\w+')
review_unclean = df2['0'].tolist()
labels = df2['label'].tolist()

print('Tokenizing ..')
review = [tokenizer.tokenize(review.lower()) for review in review_unclean]
print('Done')

# Lemmatize
nltk.download('wordnet')
reviews = []
lemmatizer = WordNetLemmatizer()
print('Lemmatizing ..')

with tqdm(total=len(review_unclean)) as pbar:
    for review in review_unclean:
        lemmatized = [lemmatizer.lemmatize(word) for word in review]
        reviews.append(lemmatized)
        pbar.update(1)
del review_unclean

# FastText Vector
vector_size = 256
window = 5

fasttext_model = 'fasttext.model'
print('Generating FastText Vectors ..')

start = time.time()
model = FastText(size=vector_size)
model.build_vocab(review)
model.train(review, window=window, min_count=1, workers=4, total_examples=model.corpus_count,
           epochs=model.epochs)
print('FastText Created in {} seconds.'.format(time.time() - start))
model.save(fasttext_model)
print('FastText Model saved at {}'.format(fasttext_model))
del model

model = FastText.load(fasttext_model)
x_vectors = model.wv
del model


# Dataset Partition

# Spliting the review1 and labels in (x_train, y_train) and (x_test, y_test) with 90% for training and 10% for testing from all the tweets.
# Maximum number of tokens allowed for each review is set to be 15.

data = review
labels = labels 

train_size = int(0.9*(len(data)))
test_size = int(0.1*(len(data)))

max_no_tokens = 15
indexes = set(np.random.choice(len(data), train_size + test_size, replace=False))
x_train = np.zeros((train_size, max_no_tokens, vector_size), dtype=K.floatx())
y_train = np.zeros((train_size, 2), dtype=np.int32)
x_test = np.zeros((test_size, max_no_tokens, vector_size), dtype=K.floatx())
y_test = np.zeros((test_size, 2), dtype=np.int32)

data_index = []
for i, index in enumerate(indexes):
    data_index.append(review_unclean[index])
    
    for t, token in enumerate(data[index]):
        if t >= max_no_tokens:
            break
        if token not in x_vectors:
            continue  
        if i < train_size:
            x_train[i, t, :] = x_vectors[token]
        else:
            x_test[i - train_size, t, :] = x_vectors[token]
    if i < train_size:
        y_train[i, :] = [1.0, 0.0] if labels[index] == 1 else [0.0, 1.0]
    else:
        y_test[i - train_size, :] = [1.0, 0.0] if labels[index] == 1 else [0.0, 1.0]

print(x_train.shape, y_test.shape)


# Neural Model
batch_size = 500
no_epochs = 20

model = Sequential()
model.add(Conv1D(32, kernel_size=3, activation='elu', padding='same',
                 input_shape=(max_no_tokens, vector_size)))
model.add(Conv1D(32, kernel_size=3, activation='elu', padding='same'))
model.add(Conv1D(32, kernel_size=3, activation='relu', padding='same'))
model.add(MaxPooling1D(pool_size=3))
model.add(Bidirectional(LSTM(512, dropout=0.2, recurrent_dropout=0.3)))
model.add(Dense(512, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='sigmoid'))
model.add(Dropout(0.25))
model.add(Dense(512, activation='sigmoid'))
model.add(Dropout(0.25))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6), metrics=['accuracy'])

tensorboard = TensorBoard(log_dir='logs/', histogram_freq=0, write_graph=True, write_images=True)
model.summary()


# Traininig
model.fit(x_train, y_train, batch_size=batch_size, shuffle=True, epochs=no_epochs,
         validation_data=(x_test, y_test), callbacks=[tensorboard, EarlyStopping(min_delta=0.001, patience=3)])


# Save the model
model.save('amazon-sentiment-fasttext.model')
model_final = model


# # Evaluate the model
print(model_final.metrics_names)
print(model_final.evaluate(x=x_test, y=y_test, batch_size=32, verbose=1))


# In[24]:


# Compare sentiment predict score to original data
predict = model_final.predict(x=x_test, batch_size=32)
np.savetxt("predict.csv", predict, delimiter=",")
predict = pd.read_csv("predict.csv", header = None)

print(predict[20:30])
print(y_test[20:30])
print(predict[20:30])
print(data_index[20+train_size:30+train_size])


# # Test Data
df_test = pd.read_csv('amazon_test.csv', sep = ';')

#check each label %
df_pos = df_test.loc[df2['label'] == 2]
df_neg = df_test.loc[df2['label'] == 1]
print(df_pos.shape)
print(df_neg.shape)

review_unclean2 = df_test['0'].tolist()
labels_test = df_test['label'].tolist()

# clean the data and convert to a list 
tokenizer = RegexpTokenizer('[a-zA-Z0-9]\w+')
review_test = [tokenizer.tokenize(review.lower()) for review in review_unclean2]
reviews2 = []
lemmatizer = WordNetLemmatizer()
print('Lemmatizing ..')

with tqdm(total=len(review_unclean2)) as pbar:
    for review in review_unclean2:
        lemmatized = [lemmatizer.lemmatize(word) for word in review_test]
        reviews2.append(lemmatized)
        pbar.update(1)
del review_unclean

# convert to fasttext
vector_size = 256
window = 5

fasttext_model = 'fasttext.model'
print('Generating FastText Vectors ..')

start = time.time()
model = FastText(size=vector_size)
model.build_vocab(review_test)
model.train(review_test, window=window, min_count=1, workers=4, total_examples=model.corpus_count,
           epochs=model.epochs)
print('FastText Created in {} seconds.'.format(time.time() - start))
model.save(fasttext_model)
print('FastText Model saved at {}'.format(fasttext_model))

del model

model = FastText.load(fasttext_model)
x_vectors = model.wv
del model


# Dataset Partition

# Make all(100%, 400000 rows) data from amazon_test.csv as train_test data to test
train_size = int(1*(len(review_test)))
max_no_tokens = 15

indexes = set(np.random.choice(len(review2), train_size + test_size, replace=False))

x_train2 = np.zeros((train_size, max_no_tokens, vector_size), dtype=K.floatx())
y_train2 = np.zeros((train_size, 2), dtype=np.int32)

for i, index in enumerate(indexes):
    for t, token in enumerate(review2[index]):
        if t >= max_no_tokens:
            break    
        if token not in x_vectors:
            continue    
        if i < train_size:
            x_train2[i, t, :] = x_vectors[token]  
    if i < train_size:
        y_train2[i, :] = [1.0, 0.0] if label2[index] == 1 else [0.0, 1.0]

print(x_train2.shape, y_train2.shape)

# Evaluate by 100% test data
model = model_final
model.evaluate(x=x_train2, y=y_train2, batch_size=32, verbose=1)

