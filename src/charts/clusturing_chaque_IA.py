import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from kneed import KneeLocator

# Téléchargements préalables des ressources NLTK (à faire une seule fois)
from nltk import download
download('punkt')
download('stopwords')

# Chargement et prétraitement des données
data = pd.read_csv('data/captions.csv')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(text):
    tokens = word_tokenize(text)
    tokens = [w.lower() for w in tokens if w.isalpha()]
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [stemmer.stem(w) for w in tokens]
    return tokens

# Préparation pour Doc2Vec
tagged_data = [TaggedDocument(words=preprocess(text), tags=[i]) for i, text in enumerate(data['caption'])]

# Assignation des couleurs par IA
ia_colors = {
    'bart_1': 'red',
    'bart_2': 'blue',
    't5_1': 'green',
    't5_2': 'purple',
    'FST': 'orange'
}

# Construction et entraînement du modèle Doc2Vec (commun à toutes les IAs)
model = Doc2Vec(vector_size=50, min_count=2, epochs=40)
model.build_vocab(tagged_data)
model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

for ia in ia_colors.keys():
    ia_data = data[data['createur'] == ia]

    # Filtre les données taggées pour l'IA en cours
    ia_tagged_data = [tagged_data[i] for i in ia_data.index]

    # Génération des vecteurs pour l'IA spécifique
    vectors = [model.dv[i] for i in ia_data.index]

    # Convertir les vecteurs en tableau NumPy pour t-SNE
    vectors_np = np.array(vectors)

    # Réduction de dimension avec t-SNE
    tsne_model = TSNE(n_components=2, random_state=42)
    tsne_vectors = tsne_model.fit_transform(vectors_np)

    # Détermination du nombre optimal de clusters
    inertia = []
    K_range = range(1, 10)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42).fit(tsne_vectors)
        inertia.append(kmeans.inertia_)

    kl = KneeLocator(K_range, inertia, curve="convex", direction="decreasing")
    optimal_clusters = kl.elbow

    # Clustering avec K-means utilisant le nombre optimal de clusters
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42).fit(tsne_vectors)
    labels = kmeans.labels_

    # Visualisation pour l'IA spécifique
    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_vectors[:, 0], tsne_vectors[:, 1], c=ia_colors[ia], label=ia, s=3, alpha=0.6)

    # Sauvegarde de l'image
    plt.title(f'Clustering of AI Generated Captions: {ia}')
    plt.legend(markerscale=2)
    plt.savefig(f'clustering_{ia}.png')
    plt.close()

    # Extraction et écriture des phrases de chaque cluster dans un fichier spécifique
    with open(f'cluster_{ia}.txt', 'w') as f:
        for i in range(optimal_clusters):
            cluster_mask = labels == i
            cluster_data = ia_data[cluster_mask]
            selected_phrases = cluster_data.sample(n=min(50, len(cluster_data)), random_state=42)['caption']
            f.write(f"Cluster {i + 1}:\n")
            for phrase in selected_phrases:
                f.write(f"Phrase: {phrase}\n")
            f.write("\n")
