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
X = X[:100]

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

# Hierarchical clustering
plt.figure(figsize=(8, 8))
plt.title('Visualizing the data')
dendrogram = shc.dendrogram(shc.linkage(tfidf_matrix.toarray(), method='ward'))
plt.show()
