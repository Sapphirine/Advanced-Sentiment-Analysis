
# coding: utf-8


# Install the latest Tensorflow version.
#!pip3 install --quiet "tensorflow>=1.7"
# Install TF-Hub.
#!pip3 install --quiet tensorflow-hub
#!pip3 install seaborn

import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
from keras import backend as K
import keras.layers as layers
from keras.models import Model
np.random.seed(10)
module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
# Import the Universal Sentence Encoder's TF Hub module
embed = hub.Module(module_url)

amazon = pd.read_csv('amazon.csv',sep=';')
#amazon =amazon.sample(frac=0.01)
#amazon.head(5)

amazon['label']-=1

from sklearn.model_selection import train_test_split
train, test = train_test_split(amazon, test_size=0.2)
print('train shape:',train.shape)
print('test shape:',test.shape)

# Training input on the whole training set with no limit on training epochs.
train_input_fn = tf.estimator.inputs.pandas_input_fn(
    train, train["label"], num_epochs=None, shuffle=True) 
# Prediction on the whole training set.
predict_train_input_fn = tf.estimator.inputs.pandas_input_fn(
    train, train["label"], shuffle=False)
# Prediction on the test set.
predict_test_input_fn = tf.estimator.inputs.pandas_input_fn(
    test, test["label"],shuffle=False)

#review tf universal sentence encoder tutorial definetely help...
#optional ?  seems yes. depends on what embedding method you want. 
# embedded_text_feature_column = hub.text_embedding_column(
#     key="0", 
#     module_spec="https://tfhub.dev/google/nnlm-en-dim128/1")  

#trainable: Whether or not the Module is trainable. False by default, 
#meaning the pre-trained weights are frozen. 
#This is different from the ordinary tf.feature_column.embedding_column(), 
#but that one is intended for training from scratch.
#mark for train_module-----trainable !!
#check how universal-sentence-encoder-large, true and false different?


# # Build model on amazon review
def train_and_evaluate_with_module(hub_module, train_module=False):  #difference between tutorial transfer learning and basic classifier is train_module =?
  embedded_text_feature_column = hub.text_embedding_column(
      key="0", module_spec=hub_module, trainable=train_module)

  estimator = tf.estimator.DNNClassifier(
      hidden_units=[500, 100],
      feature_columns=[embedded_text_feature_column],
      n_classes=2,
      optimizer=tf.train.AdagradOptimizer(learning_rate=0.003))

  estimator.train(input_fn=train_input_fn, steps=10) #eg steps=1000

  train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)
  test_eval_result = estimator.evaluate(input_fn=predict_test_input_fn)

  training_set_accuracy = train_eval_result["accuracy"]
  test_set_accuracy = test_eval_result["accuracy"]

  return {
      "Training accuracy": training_set_accuracy,
      "Test accuracy": test_set_accuracy
  }

results = {}
results["nnlm-en-dim128"] = train_and_evaluate_with_module(
    "https://tfhub.dev/google/nnlm-en-dim128/1")
results["nnlm-en-dim128-with-module-training"] = train_and_evaluate_with_module(
    "https://tfhub.dev/google/nnlm-en-dim128/1", True)
results["universal-sentence-encoder-large/3_a"] = train_and_evaluate_with_module(
    "https://tfhub.dev/google/universal-sentence-encoder-large/3", True)
results["universal-sentence-encoder-large/3_b"] = train_and_evaluate_with_module(
    "https://tfhub.dev/google/universal-sentence-encoder-large/3")

pd.DataFrame.from_dict(results, orient="index")


# # Trump part
trump = pd.read_csv('trump.csv',sep=';') #removed train
trump_train = pd.read_csv('trump_label.csv',sep=';')

trump.head(5)
trump_train['label']-=1

trump_train.head(5)

# Training input on the whole training set with no limit on training epochs.
train_input_fn = tf.estimator.inputs.pandas_input_fn(
    trump_train, trump_train["label"], num_epochs=None, shuffle=True) 
# Prediction on the whole training set.
predict_train_input_fn = tf.estimator.inputs.pandas_input_fn(
    trump_train, trump_train["label"], shuffle=False)
# Prediction on the test set.
predict_test_input_fn = tf.estimator.inputs.pandas_input_fn(
    trump,shuffle=False)

hub_module ="https://tfhub.dev/google/universal-sentence-encoder-large/3"
train_module = True
embedded_text_feature_column = hub.text_embedding_column(key="content", module_spec=hub_module, trainable=train_module)

estimator = tf.estimator.DNNClassifier(
    hidden_units=[500, 100],
    feature_columns=[embedded_text_feature_column],
    n_classes=2,
    optimizer=tf.train.AdagradOptimizer(learning_rate=0.003))

estimator.train(input_fn=train_input_fn, steps=10) #eg steps=1000, depends on train data set size

train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)
predicts = estimator.predict(input_fn=predict_test_input_fn)
predicts = list(predicts)
print(predicts[0:3])

#recoded output possibilities
value_ls = []
for i in range(len(predicts)):
    value = predicts[i]['probabilities'][0]
    value_ls.append(value)
value_ls[-10:]

len(value_ls)

#export output possibilities
df = pd.DataFrame(value_ls, columns=['predict'])
df.to_csv('predict2.csv',index=False)
train_eval_result

#Visualization Part
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
%matplotlib inline
from numpy import array

predict = pd.read_csv('predict2.csv')
trump = pd.read_csv('trump.csv', sep = ';')
trump['predict'] = predict['predict']

df_2017 = trump.loc[trump['year'] == 2017]  
df_2018 = trump.loc[trump['year'] == 2018]
print(df_2017.shape)
print(df_2018.shape)

pos_2017 = df_2017.loc[df_2017['predict'] < 0.5] 
neg_2017 = df_2017.loc[df_2017['predict'] >= 0.5]

pos_2018 = df_2018.loc[df_2018['predict'] < 0.5] 
neg_2018 = df_2018.loc[df_2018['predict'] >= 0.5]

pos_2017_by_month = pos_2017.groupby('month')['predict'].count().tolist()
neg_2017_by_month = neg_2017.groupby('month')['predict'].count().tolist()
pos_2018_by_month = pos_2018.groupby('month')['predict'].count().tolist()
neg_2018_by_month = neg_2018.groupby('month')['predict'].count().tolist()

print(pos_2017_by_month)
print(neg_2017_by_month)
print(pos_2018_by_month)
print(neg_2018_by_month)

pos_by_month = pos_2017_by_month + pos_2018_by_month
neg_by_month = neg_2017_by_month + neg_2018_by_month

X = ['201701', '201702', '201703', '201704', '201705', '201706', '201707', '201708',
    '201709', '201710', '201711', '201712', '201801', '201802', '201803', '201804',
    '201805', '201806', '201807', '201808']
Y1 = array(pos_by_month)
Y2 = array(neg_by_month)

plt.figure(figsize=(12,8))
plt.bar(X, +Y1, facecolor='#9999ff', edgecolor='white', label = 'Positive')
plt.bar(X, -Y2, facecolor='#ff9999', edgecolor='white', label = 'Negative')
plt.title('Trump\'s Twitter Sentiment 01/2017 to 08/2018', fontsize= 17)
plt.xticks(rotation=70)
#plt.legend((Y1, Y2), ('Postive', 'Negative'))

plt.xlabel("Timeline", fontsize=18)
plt.ylabel("Number of Positive/Negative Tweets", fontsize=12)

# Manually evaluate our prediction score and the real tweet content
trump[400:410]