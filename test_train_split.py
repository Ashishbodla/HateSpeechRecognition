import pandas as pd 
from sklearn.model_selection import train_test_split

df = pd.read_csv('Cleaned_text.csv',usecols=['class','clean_txt']) #import data

#Splitting into train and test
X_train, X_test, y_train, y_test = train_test_split(df['clean_txt'], df['class'], test_size=0.20, random_state=42) 

#concat and save the files
pd.concat([X_train,y_train],axis=1).to_csv('Training_data.csv')
pd.concat([X_test,y_test],axis=1).to_csv('Test_data.csv')
