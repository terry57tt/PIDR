import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from gensim.models import Doc2Vec

def ascendant_classification_by_IA(createur):
    # Download NLTK resources
    nltk.download('punkt')
    nltk.download('stopwords')

    # Read the CSV file
    X = pd.read_csv('data/captions.csv')


    # Take a random sample of 100 sentences of a specific creator
    X_sample = X[X['createur'] == createur].sample(n=100, random_state=42)

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

        preprocessed_captions.append(words)

    # Train Word2Vec model
    model = Doc2Vec(sentences=preprocessed_captions, vector_size=100, window=5, min_count=1, workers=4)

    # Get word embeddings
    word_vectors = model.wv

    # Compute sentence embeddings
    sentence_embeddings = []
    for words in preprocessed_captions:
        embeddings = [word_vectors[word] for word in words if word in word_vectors]
        if embeddings:
            sentence_embedding = sum(embeddings) / len(embeddings)
            sentence_embeddings.append(sentence_embedding)
        else:
            sentence_embeddings.append([])

    # Hierarchical clustering
    plt.figure(figsize=(8, 8))
    plt.title('Visualizing the data for the creator: ' + createur)
    
    # Combine creator's name with the id for labeling
    labels = [f"{id}_{idInterpretation_id}" for id, idInterpretation_id in zip(X_sample['id'], X_sample['idInterpretation_id'])]

    # Plot dendrogram
    dendrogram = shc.dendrogram(shc.linkage(sentence_embeddings, method='ward'), labels=labels, orientation='right')

    plt.show()

ascendant_classification_by_IA('bart_1')