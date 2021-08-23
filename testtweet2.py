import pickle
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, GRU 
from keras.layers.embeddings import Embedding 
import pandas as pd
import pickle
import numpy as np
from keras.models import load_model
mod1 = load_model('RNN_LSTM.h5')
tkobj = pickle.load(open('TOkenizer.pkl','rb'))
pred_map = {0:'Hate',1:'Offensive',2:'Neither'}

it = input('Text for RNN Model: ')
X_train_tkns = tkobj.texts_to_sequences([it])

X_train_pad = pad_sequences(X_train_tkns, maxlen=25, padding = 'post')

pred = mod1.predict(X_train_pad)
result = np.where(pred.flatten() == np.amax(pred))
print('The tweet ', it ,'is classified as', pred_map[np.where(pred.flatten() == np.amax(pred))[0][0]])
