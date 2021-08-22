import pandas as pd
import re, string
import nltk
#nltk.download()
from nltk.corpus import stopwords

def clean_text(text):
    global stopword
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('@[^>]+:','',text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text

df = pd.read_csv('./Data_files/labeled_data.csv', usecols=['class','tweet'])
stemmer = nltk.SnowballStemmer("english")
stopword=set(stopwords.words('english'))
df['clean_txt'] = df['tweet'].apply(lambda x: clean_text(x))
df.to_csv('Cleaned_text.csv',index=False)
