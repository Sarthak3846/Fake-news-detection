import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

real_news = pd.read_csv("C:\\Users\\Sarthak Tyagi\\Downloads\\True.csv")
fake_news = pd.read_csv("C:\\Users\\Sarthak Tyagi\\Downloads\\Fake.csv")

real_news['label']=0
fake_news['label']=1

df = pd.concat([fake_news,real_news], ignore_index=True)
df

from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

df['content']= df['title']+" "+df['text']
df['content']=df['content'].str.lower()

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['content']).toarray()
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = MultinomialNB()
model.fit(X_train,y_train)

from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test)
print(accuracy_score(y_test,y_pred))

import joblib 

joblib.dump(model,'fake_news_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')