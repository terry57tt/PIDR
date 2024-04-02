import csv
from collections import Counter
from nltk.util import ngrams

def tokenize_phrase(phrase):
    # Similaire à la fonction de tokenisation fournie précédemment
    tokens = phrase.split()  # Utilisez une tokenisation simple basée sur les espaces
    return tokens

def get_n_grams(tokens, n):
    return [' '.join(gram) for gram in ngrams(tokens, n)]

def analyze_model(captions):
    n_gram_frequencies = {n: Counter() for n in range(2, 30)}
    
    for caption in captions:
        tokens = tokenize_phrase(caption)
        for n in range(2, 21):
            n_gram_frequencies[n].update(get_n_grams(tokens, n))
    
    return n_gram_frequencies

def read_captions(filename, model_name):
    captions = []
    with open(filename, 'r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            if row['createur'] == model_name:
                captions.append(row['caption'])
    return captions

def save_results(model_name, n_gram_frequencies):
    with open(f'results_{model_name}.txt', 'w', encoding='utf-8') as file:
        for n, frequencies in n_gram_frequencies.items():
            file.write(f'{model_name} :\nTaille {n} : {len(frequencies)} n-grammes différents\n')
            for n_gram, count in frequencies.most_common(20):
                file.write(f'[{n_gram}]   [vu {count} fois]\n')
            file.write('\n')

models = ['bart_1', 'bart_2', 't5_1', 't5_2', 'FST']

for model in models:
    captions = read_captions('data/captions.csv', model)
    n_gram_frequencies = analyze_model(captions)
    save_results(model, n_gram_frequencies)
