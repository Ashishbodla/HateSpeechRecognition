#To test tf-idf Vectorizer 
# To test models on count vectorizer
import pickle
import numpy as np
import pandas as pd 

t_vec = pickle.load(open('TFIDF_vectorizer.sav','rb'))
with open('Count_vector_indices.npy','rb') as f:
    ind = np.load(f,allow_pickle=True)

pred_map = {0:'Hate',1:'Offensive',2:'Neither'}

c_vec = pickle.load(open('Count_Vectorizer.sav','rb'))
with open('Count_vector_indices.npy','rb') as f:
    ind = np.load(f,allow_pickle=True)

input_text = input('Test tweet: ')
x1 = c_vec.transform(np.array([input_text]))
df1 = pd.DataFrame(x1.todense(),columns=word_vec.get_feature_names())
df = df1[ind]
#Import a model to be tested
x2 = t_vec.transform(df)
model = pickle.load(open('./tfidf_vectors_models/MNB_counts.sav','rb'))
print('The tweet: ', input_text, '- is ', pred_map[model.predict(x2)[0]])


from keras.models import load_model
mod1 = load_model('RNN_LSTM.h5')
tok = pickle.load(open('TOkenizer.pkl','rb'))

it = input('Text for RNN Model: ')
X_train_tkns = tkobj.texts_to_sequences([it])

X_train_pad = pad_sequences(X_train_tkns, maxlen=25, padding = 'post')

pred = mod1.predict(X_train_pad)
result = np.where(pred.flatten() == np.amax(pred))
print('The tweet ', it ,'is classified as', pred_map[np.where(pred.flatten() == np.amax(pred))[0][0]])
