import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

train = pd.read_csv('Training_data.csv')
train.dropna(subset=['clean_txt'], inplace=True)

test = pd.read_csv('Test_data.csv')
test.dropna(subset=['clean_txt'], inplace=True)

#Vectorizer class initiation
word_vec = CountVectorizer(ngram_range=(1,2), analyzer='word')

#Fitting and transforming the train data

mat_tr = word_vec.fit_transform(train['clean_txt'])

#processing the vectorizer output
#finding the frequency of words in the train data
freq = sum(mat_tr).toarray()[0]
df= pd.DataFrame(freq, index=word_vec.get_feature_names(), columns=['frequency'])
df1= pd.DataFrame(mat_tr.todense(), columns=[word_vec.get_feature_names()])
ind = df[(df['frequency']>5) & (df['frequency']<50)].index #limiting the frequency
df2 = df1[np.array(ind)] #taking only the required terms based on frequency filtering
df2.reset_index(drop=True, inplace=True)
train_data_final = pd.concat([df2,train[['class','Hate','Offensive','Neither']]],axis=1)
train_data_final.to_csv('CountVectorised_train.csv')

#Test data processing steps same as input text processing

mat_ts = word_vec.transform(test['clean_txt'])
df1_ts = pd.DataFrame(mat_ts.todense(), columns=[word_vec.get_feature_names()])
df2_ts = df1_ts[np.array(ind)]
df2_ts.reset_index(drop=True)
test_data_final = pd.concat([df2_ts,test[['class','Hate','Offensive','Neither']]], axis=1)
test_data_final.to_csv('CountVectorised_test.csv')
