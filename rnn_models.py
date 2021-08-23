
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, GRU 
from keras.layers.embeddings import Embedding 
import pandas as import pd 

train = pd.read_csv('./Data_files/Training_data.csv')
test = pd.read_csv('./Data_files/Test_data.csv')
train.dropna(inplace=True)
test.dropna(inplace=True)
X_train = train.drop(columns = ['class','Hate','Offensive','Neither'])
y_train = train[['Hate','Offensive','Neither']]
X_test = test.drop(columns = ['class','Hate','Offensive','Neither'])
y_test = test[['Hate','Offensive','Neither']]

tkobj = Tokenizer()
all_ = pd.concat([train,test],axis=0)
reviews = all_['clean_txt']
tkobj.fit_on_texts(reviews)
max_length = max([len(s.split()) for s in reviews])
pickle.dump(tkobj,open('Tokenizer.pkl','wb'))
X_train_tkns = tkobj.texts_to_sequences(X_train['clean_txt'])
X_test_tkns = tkobj.texts_to_sequences(X_test['clean_txt'])
X_train_pad = pad_sequences(X_train_tkns, maxlen=max_length, padding = 'post')
X_test_pad = pad_sequences(X_test_tkns, maxlen=max_length, padding = 'post')

# RNN + GRU Modelling 
vsize = len(tkobj.word_index)+1
Embed_dim = 100
model1 = Sequential()
model1.add(Embedding(vsize,Embed_dim,input_length=max_length))
model1.add(GRU(units=32,dropout=0.2, recurrent_dropout=0.20))
model1.add(Dense(3,activation='softmax'))

model1.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

model1.fit(X_train_pad,y_train,batch_size=256,validation_data=(X_test_pad,y_test), epochs=20)
rnn_gru_preds = model1.predict(X_test_pad)
model1.save('RNN_GRU.h5')

#RNN +LSTM Modelling
model2 = Sequential()
model2.add(Embedding(vsize,Embed_dim,input_length=max_length))
model2.add(LSTM(units=32,dropout=0.2, recurrent_dropout=0.20))
model2.add(Dense(3,activation='softmax'))

model2.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
model2.fit(X_train_pad,y_train,batch_size=256,validation_data=(X_test_pad,y_test), epochs=20)
rnn_gru_preds = model2.predict(X_test_pad)
model2.save('RNN_LSTM.h5')

