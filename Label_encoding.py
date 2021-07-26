import pandas as pd

#Importing the data files from train test split output

train = pd.read_csv('Training_data.csv')
test = pd.read_csv('Test_data.csv')

#Creating dummies i.e. one hot encoding and concatenating to the original file
train = pd.concat([train,pd.get_dummies(train['class'])], axis=1)
test = pd.concat([test,pd.get_dummies(test['class']], axis=1)

#Renaming the dummy columns
train.rename(columns={0:'Hate',1:'Offensive',2:'Neither'}, inplace=True)
test.rename(columns={0:'Hate',1:'Offensive',2:'Neither'}, inplace=True)

#Dropping the unnecessary column
train.drop(columns=['Unnamed: 0'], inplace = True)
test.drop(columns=['Unnamed: 0'], inplace = True)

#Saving the files
train.to_csv('Training_data.csv', index=False)
test.to_csv('Training_data.csv', index=False)
