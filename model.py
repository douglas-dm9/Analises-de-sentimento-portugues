import pickle
import nltk
#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('rslp')
from unidecode import unidecode
import pandas as pd

stop_words = pd.read_csv('stopwords.csv',sep=';')
stop_words = set(stop_words['words'])
 
model_lr = pickle.load(open('model_lr.pkl', 'rb'))

def pre_processor(text):
  text = text.lower()
  stemmer = nltk.stem.RSLPStemmer()
  lemmatizer = nltk.stem.WordNetLemmatizer()
  tokenized = nltk.word_tokenize(text)
  filtered_sentence = [ unidecode(word) for word in tokenized if word not in stop_words ]
  #filtered_sentence = [lemmatizer.lemmatize(unidecode(word)) for word in tokenized if word not in stop_words]
  return ' '.join(c for c in filtered_sentence if c.isalpha())

def predict_sentiment(text):
    #if model_lr[1].predict_proba(model_lr[0].transform([row['text']]))[0][0] >=0.65:
    if model_lr.predict_proba([pre_processor(text)])[0][0] >= 0.65:
        return 'Negativo'
    #elif  model_lr[1].predict_proba(model_lr[0].transform([row['text']]))[0][0] <=0.35:
    elif model_lr.predict_proba([pre_processor(text)])[0][0] <= 0.35:
        return 'Positivo'
    else:
        return 'Neutro'



print(model_lr.predict_proba([pre_processor('eu amo vc')])[0][0])
#model_lr = pickle.load(open('model_lr.pkl', 'rb'))
#print(model_lr.predict_proba([pre_processor('teste')])[0][0])
