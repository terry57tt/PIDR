import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Read the CSV file
X = pd.read_csv('data/captions.csv')

# Take a random sample of 100 sentences
X_sample = X.sample(n=100, random_state=42)

# Preprocess each caption
preprocessed_captions = []
for sentence in X_sample['caption']:
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

# Hierarchical clustering
plt.figure(figsize=(8, 8))
plt.title('Visualizing the data')

# Combine creator's name with the id for labeling
labels = [f"{id} - {creator}" for id, creator in zip( X_sample['id'], X_sample['createur'])]

# Plot dendrogram
dendrogram = shc.dendrogram(shc.linkage(tfidf_matrix.toarray(), method='ward'), labels=labels, orientation='right')

plt.show()
