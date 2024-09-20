
# sentiment_analysis.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Télécharger un dataset de sentiment (par exemple des avis de films)
url = 'https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv'
data = pd.read_csv(url)

# Afficher les premières lignes pour comprendre la structure des données
print(data.head())

# Préparer les données
X = data['tweet']  # Texte des tweets
y = data['label']  # Labels des sentiments (0 = négatif, 1 = positif)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convertir le texte en vecteurs
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)

# Entraîner le modèle Naïve Bayes
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# Prédire les sentiments pour les données de test
X_test_vectorized = vectorizer.transform(X_test)
y_pred = model.predict(X_test_vectorized)

# Évaluer le modèle
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

