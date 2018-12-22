
# coding: utf-8

import findspark 
findspark.init()
from pyspark.sql import SparkSession
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, NGram, HashingTF, IDF
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType, DoubleType
from sklearn.linear_model import LogisticRegression
from langdetect import detect 
import pandas as pd
import numpy as np

# read csv files
df_train = pd.read_csv('amazon.csv', sep = ';')
df_test = pd.read_csv('amazon_test.csv', sep = ';')
print(df_train.shape)
print(df_test.shape)

# determine whether dataframe contain text 
def chooseText (df):
    series = pd.Series(df['0'])
    boolean = series.str.contains('[a-zA-Z]', regex=True)
    df['logic'] = boolean
    df = df[df['logic'] == True]
    return df.iloc[:,0:2]  

df_train = chooseText(df_train)
df_test = chooseText(df_test)
print(df_train.shape)
print(df_test.shape)

# select only English reviews
idx1 = df_train['0'].apply(lambda x: detect(x) == 'en')

# choose logic column containing true value
df_train['logic'] = idx1
df_train = df_train[df_train['logic'] == True]
df_train = df_train[['0','label']]

# change column names
df_train.columns = ['text', 'label']
train_x = pd.DataFrame(df_train['text'])
train_y = pd.DataFrame(df_train['label'])

# save the output
train_x.to_csv('train_x.csv', index = False)
train_y.to_csv('train_y.csv', index = False)

# select only english reviews
idx2 = df_test['0'].apply(lambda x: detect(x) == 'en')

# choose logic column containing true value
df_test['logic'] = idx2
df_test = df_test[df_test['logic'] == True]
df_test = df_test[['0','label']]

# change column names
df_test.columns = ['text', 'label']
test_x = pd.DataFrame(df_test['text'])
test_y = pd.DataFrame(df_test['label'])

# save the output
test_x.to_csv('test_x.csv', index = False)
test_y.to_csv('test_y.csv', index = False)

# read preprocessed Amazon dataset
spark = SparkSession.builder.appName("FinalProject").getOrCreate()
df_train_x = spark.read.csv("train_x.csv", header=True)
df_train_y = spark.read.csv("train_y.csv", header=True)
df_test_x = spark.read.csv("test_x.csv", header=True)
df_test_y = spark.read.csv("test_y.csv", header=True)
print((df_train_x.count(),len(df_train_x.columns)))
print((df_train_y.count(),len(df_train_y.columns)))
print((df_test_x.count(),len(df_test_x.columns)))
print((df_test_y.count(),len(df_test_y.columns)))

# function to get feature data
def get_feature (dataframe = df_train_x, nFeature = 200):
    # convert the input string to lowercase and then split it by regex pattern
    regexTokenizer = RegexTokenizer(inputCol="text", outputCol="words", pattern="\\W")
    words_data = regexTokenizer.transform(dataframe)
    #count_tokens = udf(lambda words: len(words), IntegerType()) # count the number of words in each review
    #words_data.select("words").withColumn("tokens", count_tokens(col("words"))).show(5,truncate=True)
    
    # remove stop words (e.g the, who, which, at, on, I)
    stopWordsRemover = StopWordsRemover(inputCol="words", outputCol="words_removed")
    words_removed_data = stopWordsRemover.transform(words_data)
    #count_tokens_new = udf(lambda words_removed: len(words_removed), IntegerType())
    #words_removed_data.select("words_removed").withColumn("tokens_new", count_tokens_new(col("words_removed"))).show(5,truncate=True)
    
    # transform input features into n-grams
    #nGram = NGram(n=2, inputCol="words_removed", outputCol="ngrams")
    #ngrams_data = nGram.transform(words_removed_data)
    
    # transform list of words to words frequency vectors
    hashingTF = HashingTF(inputCol="words_removed", outputCol="words_freq", numFeatures=nFeature)
    words_freq_data = hashingTF.transform(words_removed_data)
    #words_freq_data.select("words_freq").show(5,truncate=True)
    
    # compute the IDF vector and scale words frequencies by IDF
    idf = IDF(inputCol="words_freq", outputCol="features")
    idf_model = idf.fit(words_freq_data)
    feature_data = idf_model.transform(words_freq_data).select("features") 
    
    return feature_data

feature_train =  get_feature(dataframe = df_train_x, nFeature = 200)
print((feature_train.count(),len(feature_train.columns)))
feature_train.take(1)

feature_test =  get_feature(dataframe = df_test_x, nFeature = 200)
print((feature_test.count(),len(feature_test.columns)))
feature_test.take(1)

train_x = feature_train.toPandas()
train_y = df_train_y.toPandas()

test_x = feature_test.toPandas()
test_y = df_test_y.toPandas()

series1 = train_x['features'].apply(lambda x : np.array(x.toArray())).as_matrix().reshape(-1,1)
train_x = np.apply_along_axis(lambda x : x[0], 1, series1)
train_y = train_y['label'].values.reshape(-1,1).ravel()

series2 = test_x['features'].apply(lambda x : np.array(x.toArray())).as_matrix().reshape(-1,1)
test_x = np.apply_along_axis(lambda x : x[0], 1, series2)
test_y = test_y['label'].values.reshape(-1,1).ravel()

# build logistic regression model 
lr = LogisticRegression()
lrm = lr.fit(train_x, train_y)
lrm.score(train_x, train_y)

# get test accuracy
label_pred = lrm.predict(test_x)
np.mean(label_pred == test_y)

