import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from gensim.models import Word2Vec


X = pd.read_csv('data/captions.csv')
ids = X['idInterpretation_id'].unique()
X_sample = pd.DataFrame(columns=X.columns)
for i in range(0,25):
    x = np.random.choice(ids)
    new_sample = X[X['idInterpretation_id'] == x]
    if X_sample.empty:
        X_sample = new_sample
    else:
        X_sample = pd.concat([X_sample,new_sample])       

# Download NLTK resources
nltk.download('punkt',quiet=True)
nltk.download('stopwords',quiet=True)

# Preprocess each caption
preprocessed_captions = []
for index, sentence in X_sample['caption'].items():
    # Tokenization
    words = word_tokenize(sentence)
    # Lowercasing and removing non-alphabetic words
    words = [word.lower() for word in words if word.isalpha()]
    # Removing stopwords
    stop_words = stopwords.words('english') 
    words = [word for word in words if word not in stop_words]

    # Stemming
    stemmer = SnowballStemmer('english')
    words = [stemmer.stem(word) for word in words]

    preprocessed_captions.append(words)

# Train Doc2Vec model
model = Word2Vec(vector_size=100, window=5, min_count=1, workers=4)
model.build_vocab(preprocessed_captions)
model.train(preprocessed_captions, total_examples=model.corpus_count, epochs=10)

# Compute sentence embeddings
sentence_embeddings = []
for words in preprocessed_captions:
    embeddings = [model.wv[word] for word in words if word in model.wv.key_to_index]
    if embeddings:
        sentence_embedding = sum(embeddings) / len(embeddings)
        sentence_embeddings.append(sentence_embedding)
    else:
        sentence_embeddings.append([])

# Hierarchical clustering
plt.figure(figsize=(8, 8))
plt.title('Ascendant classification')
labels = X_sample['createur'].values

dend = shc.dendrogram(shc.linkage(sentence_embeddings, method='ward'), labels=labels, orientation='right')

plt.savefig('ACH/ascendant_classification.png')
plt.show()