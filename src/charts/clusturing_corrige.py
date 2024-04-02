# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from gensim.models.doc2vec import Doc2Vec, TaggedDocument
# from sklearn.manifold import TSNE
# from sklearn.cluster import KMeans
# from nltk import download
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer
# from kneed import KneeLocator

# download('punkt')
# download('stopwords')

# # Chargement et prétraitement des données
# data = pd.read_csv('data/captions.csv')
# stop_words = set(stopwords.words('english'))
# stemmer = PorterStemmer()

# def preprocess(text):
#     tokens = word_tokenize(text)
#     tokens = [w.lower() for w in tokens if w.isalpha()]
#     tokens = [w for w in tokens if not w in stop_words]
#     tokens = [stemmer.stem(w) for w in tokens]
#     return tokens

# # Préparation pour Doc2Vec
# tagged_data = [TaggedDocument(words=preprocess(text), tags=[i]) for i, text in enumerate(data['caption'])]

# # Construction du modèle Doc2Vec
# model = Doc2Vec(vector_size=50, min_count=2, epochs=40)
# model.build_vocab(tagged_data)
# model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

# # Génération des vecteurs
# vectors = [model.dv[i] for i in range(len(tagged_data))]

# # Convertir les vecteurs en tableau NumPy pour t-SNE
# vectors_np = np.array(vectors)

# # Réduction de dimension avec t-SNE
# tsne_model = TSNE(n_components=2, random_state=42)
# tsne_vectors = tsne_model.fit_transform(vectors_np)

# # Détermination du nombre optimal de clusters
# inertia = []
# K_range = range(1, 10)
# for k in K_range:
#     kmeans = KMeans(n_clusters=k, random_state=42).fit(tsne_vectors)
#     inertia.append(kmeans.inertia_)

# kl = KneeLocator(K_range, inertia, curve="convex", direction="decreasing")
# optimal_clusters = kl.elbow

# # Clustering avec K-means utilisant le nombre optimal de clusters
# kmeans = KMeans(n_clusters=optimal_clusters, random_state=42).fit(tsne_vectors)

# # Visualisation
# plt.figure(figsize=(10, 8))
# colors = ['red', 'green', 'blue', 'purple', 'orange']
# for i in range(optimal_clusters):
#     mask = kmeans.labels_ == i
#     plt.scatter(tsne_vectors[mask, 0], tsne_vectors[mask, 1], color=colors[i % len(colors)], label=f'Cluster {i+1}')

# plt.title('Clustering of AI Generated Captions with Doc2Vec & t-SNE')
# plt.legend()
# plt.savefig('plot/doc2vec_tsne_kmeans_clustering.png')
# plt.show()

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from gensim.models.doc2vec import Doc2Vec, TaggedDocument
# from sklearn.manifold import TSNE
# from sklearn.cluster import KMeans
# from nltk import download
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer
# from kneed import KneeLocator

# download('punkt')
# download('stopwords')

# # Chargement et prétraitement des données
# data = pd.read_csv('data/captions.csv')
# stop_words = set(stopwords.words('english'))
# stemmer = PorterStemmer()

# def preprocess(text):
#     tokens = word_tokenize(text)
#     tokens = [w.lower() for w in tokens if w.isalpha()]
#     tokens = [w for w in tokens if not w in stop_words]
#     tokens = [stemmer.stem(w) for w in tokens]
#     return tokens

# # Préparation pour Doc2Vec
# tagged_data = [TaggedDocument(words=preprocess(text), tags=[i]) for i, text in enumerate(data['caption'])]

# # Construction du modèle Doc2Vec
# model = Doc2Vec(vector_size=50, min_count=2, epochs=40)
# model.build_vocab(tagged_data)
# model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

# # Génération des vecteurs
# vectors = [model.dv[i] for i in range(len(tagged_data))]

# # Convertir les vecteurs en tableau NumPy pour t-SNE
# vectors_np = np.array(vectors)

# # Réduction de dimension avec t-SNE
# tsne_model = TSNE(n_components=2, random_state=42)
# tsne_vectors = tsne_model.fit_transform(vectors_np)

# # Détermination du nombre optimal de clusters
# inertia = []
# K_range = range(1, 10)
# for k in K_range:
#     kmeans = KMeans(n_clusters=k, random_state=42).fit(tsne_vectors)
#     inertia.append(kmeans.inertia_)

# kl = KneeLocator(K_range, inertia, curve="convex", direction="decreasing")
# optimal_clusters = kl.elbow

# # Clustering avec K-means utilisant le nombre optimal de clusters
# kmeans = KMeans(n_clusters=optimal_clusters, random_state=42).fit(tsne_vectors)
# labels = kmeans.labels_

# # Assignation des couleurs par IA
# ia_colors = {
#     'bart_1': 'red',
#     'bart_2': 'blue',
#     't5_1': 'green',
#     't5_2': 'purple',
#     'FST': 'orange'
# }
# data['color'] = data['createur'].apply(lambda x: ia_colors.get(x, 'black'))

# # Visualisation avec des points plus petits et entourés
# plt.figure(figsize=(10, 8))
# for ia, color in ia_colors.items():
#     ia_mask = data['createur'] == ia
#     plt.scatter(tsne_vectors[ia_mask, 0], tsne_vectors[ia_mask, 1], c=color, label=ia, s=10, alpha=0.6)

# # Dessin des cercles autour des clusters
# for center in kmeans.cluster_centers_:
#     circle = plt.Circle(center, radius=2, color='gray', fill=False, linewidth=2, alpha=0.5)
#     plt.gca().add_patch(circle)

# plt.title('Clustering of AI Generated Captions with Doc2Vec & t-SNE')
# plt.legend(markerscale=2)
# plt.savefig('plot/doc2vec_tsne_kmeans_clustering.png')
# plt.close()

# # Extraction et écriture des phrases de chaque cluster dans un fichier
# with open('clusters.txt', 'w') as f:
#     for i in range(optimal_clusters):
#         cluster_mask = labels == i
#         f.write(f"Cluster {i + 1}:\n")
#         for caption, creator in zip(data[cluster_mask]['caption'].head(40), data[cluster_mask]['createur'].head(40)):
#             f.write(f"Phrase: {caption}   [IA: {creator}]\n")
#         f.write("\n")

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from gensim.models.doc2vec import Doc2Vec, TaggedDocument
# from sklearn.manifold import TSNE
# from sklearn.cluster import KMeans
# from nltk import download
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer
# from kneed import KneeLocator

# download('punkt')
# download('stopwords')

# # Chargement et prétraitement des données
# data = pd.read_csv('data/captions.csv')
# stop_words = set(stopwords.words('english'))
# stemmer = PorterStemmer()

# def preprocess(text):
#     tokens = word_tokenize(text)
#     tokens = [w.lower() for w in tokens if w.isalpha()]
#     tokens = [w for w in tokens if not w in stop_words]
#     tokens = [stemmer.stem(w) for w in tokens]
#     return tokens

# # Préparation pour Doc2Vec
# tagged_data = [TaggedDocument(words=preprocess(text), tags=[i]) for i, text in enumerate(data['caption'])]

# # Construction du modèle Doc2Vec
# model = Doc2Vec(vector_size=50, min_count=2, epochs=40)
# model.build_vocab(tagged_data)
# model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

# # Génération des vecteurs
# vectors = [model.dv[i] for i in range(len(tagged_data))]

# # Convertir les vecteurs en tableau NumPy pour t-SNE
# vectors_np = np.array(vectors)

# # Réduction de dimension avec t-SNE
# tsne_model = TSNE(n_components=2, random_state=42)
# tsne_vectors = tsne_model.fit_transform(vectors_np)

# # Détermination du nombre optimal de clusters
# inertia = []
# K_range = range(1, 10)
# for k in K_range:
#     kmeans = KMeans(n_clusters=k, random_state=42).fit(tsne_vectors)
#     inertia.append(kmeans.inertia_)

# kl = KneeLocator(K_range, inertia, curve="convex", direction="decreasing")
# optimal_clusters = kl.elbow

# # Clustering avec K-means utilisant le nombre optimal de clusters
# kmeans = KMeans(n_clusters=optimal_clusters, random_state=42).fit(tsne_vectors)
# labels = kmeans.labels_

# # Assignation des couleurs par IA
# ia_colors = {
#     'bart_1': 'red',
#     'bart_2': 'blue',
#     't5_1': 'green',
#     't5_2': 'purple',
#     'FST': 'orange'
# }
# data['color'] = data['createur'].apply(lambda x: ia_colors.get(x, 'black'))

# # Visualisation avec des points plus petits et entourés
# plt.figure(figsize=(10, 8))
# for ia, color in ia_colors.items():
#     ia_mask = data['createur'] == ia
#     plt.scatter(tsne_vectors[ia_mask, 0], tsne_vectors[ia_mask, 1], c=color, label=ia, s=3, alpha=0.6)

# # Dessin des cercles autour des clusters
# for i, center in enumerate(kmeans.cluster_centers_):
#     # Calcul du rayon du cercle à dessiner
#     points_in_cluster = tsne_vectors[labels == i]
#     radius = np.max(np.sqrt(np.sum((points_in_cluster - center) ** 2, axis=1)))
#     circle = plt.Circle(center, radius, color='black', fill=False, linewidth=1.5, alpha=0.7)
#     plt.gca().add_patch(circle)

# plt.title('Clustering of AI Generated Captions with Doc2Vec & t-SNE')
# plt.legend(markerscale=2)
# plt.savefig('plot/doc2vec_tsne_kmeans_clustering.png')
# plt.close()

# # Extraction et écriture des phrases de chaque cluster dans un fichier
# with open('clusters.txt', 'w') as f:
#     for i in range(optimal_clusters):
#         cluster_mask = labels == i
#         f.write(f"Cluster {i + 1}:\n")
#         for caption, creator in zip(data[cluster_mask]['caption'].head(40), data[cluster_mask]['createur'].head(40)):
#             f.write(f"Phrase: {caption}   [IA: {creator}]\n")
#         f.write("\n")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from nltk import download
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from kneed import KneeLocator

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

# Construction du modèle Doc2Vec
model = Doc2Vec(vector_size=50, min_count=2, epochs=40)
model.build_vocab(tagged_data)
model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

# Génération des vecteurs
vectors = [model.dv[i] for i in range(len(tagged_data))]

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

# Création d'une liste de phrases pour chaque cluster
cluster_phrases = [[] for _ in range(optimal_clusters)]

# Assignation des couleurs par IA
ia_colors = {
    'bart_1': 'red',
    'bart_2': 'blue',
    't5_1': 'green',
    't5_2': 'purple',
    'FST': 'orange'
}
data['color'] = data['createur'].apply(lambda x: ia_colors.get(x, 'black'))

# Visualisation avec des points plus petits et entourés
plt.figure(figsize=(10, 8))
for ia, color in ia_colors.items():
    ia_mask = data['createur'] == ia
    plt.scatter(tsne_vectors[ia_mask, 0], tsne_vectors[ia_mask, 1], c=color, label=ia, s=3, alpha=0.6)

# Dessin des cercles autour des clusters
for i, center in enumerate(kmeans.cluster_centers_):
    # Calcul du rayon du cercle à dessiner
    points_in_cluster = tsne_vectors[labels == i]
    radius = np.max(np.sqrt(np.sum((points_in_cluster - center) ** 2, axis=1)))
    circle = plt.Circle(center, radius, color='black', fill=False, linewidth=1.5, alpha=0.7)
    plt.gca().add_patch(circle)

plt.title('Clustering of AI Generated Captions with Doc2Vec & t-SNE')
plt.legend(markerscale=2)
plt.savefig('plot/doc2vec_tsne_kmeans_clustering2.png')
plt.close()

# Extraction et écriture des phrases de chaque cluster dans un fichier
with open('clusters2.txt', 'w') as f:
    for i in range(optimal_clusters):
        cluster_mask = labels == i
        # Récupération des phrases du cluster
        cluster_data = data[cluster_mask]
        # Sélection aléatoire de 50 phrases
        selected_phrases = cluster_data.sample(n=50)['caption']
        # Écriture des phrases sélectionnées dans le fichier
        f.write(f"Cluster {i + 1}:\n")
        for phrase in selected_phrases:
            f.write(f"Phrase: {phrase}\n")
        f.write("\n")
