import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Read the CSV file
X = pd.read_csv('data/captions.csv')

# Preprocess each caption
preprocessed_captions = []
for sentence in X['caption']:
    # Tokenization
    words = word_tokenize(sentence)

    # Lowercasing and removing non-alphabetic words
    words = [word.lower() for word in words if word.isalpha()]

    # Removing stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]

    # Joining the preprocessed words back into a sentence
    preprocessed_sentence = ' '.join(words)
    preprocessed_captions.append(preprocessed_sentence)

# Initialize TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the preprocessed text
tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_captions)

# Choose the number of clusters
num_clusters = 4

ia_colors = {
    'bart_1': 'red',
    'bart_2': 'blue',
    't5_1': 'green',
    't5_2': 'purple',
    'FST': 'orange'
}

# Ajouter une colonne pour la couleur correspondant à chaque IA
X['color'] = X['createur'].apply(lambda ia: ia_colors[ia])

# Visualisation avec PCA
pca = PCA(n_components=2)
reduced_tfidf_matrix = pca.fit_transform(tfidf_matrix.toarray())

# Effectuer le clustering K-means
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(reduced_tfidf_matrix)

# Créer une figure pour le scatter plot
plt.figure(figsize=(10, 8))
plt.title('K-Means Clustering of AI Generated Sentences')

# Plot the points with specific colors for each IA and draw circles around clusters
for ia, color in ia_colors.items():
    # Select only the data for this IA
    ia_data = reduced_tfidf_matrix[X['createur'] == ia]
    ia_labels = kmeans.labels_[X['createur'] == ia]

    # Plot the data points
    plt.scatter(ia_data[:, 0], ia_data[:, 1], c=color, label=ia, s=3, alpha=0.5)

# Draw circles around the clusters
for i, center in enumerate(kmeans.cluster_centers_):
    # Calculate the radius of the circle to be drawn
    points_in_cluster = reduced_tfidf_matrix[kmeans.labels_ == i]
    radius = np.max(np.sqrt(np.sum((points_in_cluster - center) ** 2, axis=1)))
    circle = plt.Circle(center, radius, color='black', fill=False, linewidth=1.5, alpha=0.7)
    plt.gca().add_patch(circle)

# Adjust legend and axis labels
plt.legend(loc='upper right')
plt.xlabel('PCA Feature 1')
plt.ylabel('PCA Feature 2')

# Save the plot as a PNG image
plt.savefig('plot/kmeans_clustering.png')
plt.close()

labels = kmeans.labels_

# Extraction et écriture des phrases de chaque cluster dans un fichier
with open('clusters_tidf.txt', 'w') as f:
    for i in range(3):
        cluster_mask = labels == i
        # Récupération des phrases du cluster
        cluster_data = X[cluster_mask]
        # Sélection aléatoire de 50 phrases
        selected_phrases = cluster_data.sample(n=50)['caption']
        # Écriture des phrases sélectionnées dans le fichier
        f.write(f"Cluster {i + 1}:\n")
        for phrase in selected_phrases:
            f.write(f"Phrase: {phrase}\n")
        f.write("\n")