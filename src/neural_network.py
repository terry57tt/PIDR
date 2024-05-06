import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from keras.layers import Input, Dense
from keras.models import Model
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Charger les données depuis le fichier CSV
X = pd.read_csv('data/captions.csv')

# Prétraitement des légendes
preprocessed_captions = []
for sentence in X['caption']:
    # Tokenization
    words = word_tokenize(sentence)

    # Conversion en minuscules et suppression des mots non alphabétiques
    words = [word.lower() for word in words if word.isalpha()]

    # Suppression des mots vides
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    preprocessed_captions.append(words)

# Entraînement du modèle Word2Vec
model = Word2Vec(sentences=preprocessed_captions, vector_size=100, window=5, min_count=1, workers=4)

# Vectorisation des textes
vecteurs_textes = []
for texte in X['caption']:
    vecteur_moyen = np.mean([model.wv[mot] for mot in texte.split() if mot in model.wv.key_to_index], axis=0)
    vecteurs_textes.append(vecteur_moyen)

vecteurs_textes = np.array(vecteurs_textes)

# Création du réseau de neurones auto-encodeur
entree = Input(shape=(100,))
code = Dense(32, activation='relu')(entree)
sortie = Dense(100, activation='sigmoid')(code)

auto_encodeur = Model(entree, sortie)
encodeur = Model(entree, code)

# Compilation du réseau de neurones
auto_encodeur.compile(optimizer='adam', loss='mean_squared_error')

# Entraînement du réseau de neurones
auto_encodeur.fit(vecteurs_textes, vecteurs_textes, epochs=100, batch_size=32, shuffle=True)

# Calcul de la distance entre les vecteurs codés pour évaluer la diversité expressive
vecteurs_codes = encodeur.predict(vecteurs_textes)
distances = []
for i in range(len(vecteurs_codes)):
    for j in range(i + 1, len(vecteurs_codes)):
        distance = np.linalg.norm(vecteurs_codes[i] - vecteurs_codes[j])
        distances.append(distance)

diversite_expressive = np.mean(distances)
print("Diversité expressive :", diversite_expressive)
