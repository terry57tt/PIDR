import csv
import random

import matplotlib.pyplot as plt
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score
from rouge_score import rouge_scorer
import data_parser
import pandas as pd

def tokenize_phrase(phrase):
    tokens = []
    current_token = ""
    for char in phrase:
        if char in ("#", "&"):
            char = " "
        current_token += char
    tokens = current_token.split()
    return tokens

def extract_sentences_by_creator():
    data = pd.read_csv('data/captions.csv')

    sentences_by_creator = {
        'bart_1': data[data['createur'] == 'bart_1']['caption'].tolist(),
        'bart_2': data[data['createur'] == 'bart_2']['caption'].tolist(),
        't5_1': data[data['createur'] == 't5_1']['caption'].tolist(),
        't5_2': data[data['createur'] == 't5_2']['caption'].tolist(),
        'FST': data[data['createur'] == 'FST']['caption'].tolist()
    }

    return sentences_by_creator

def create_diversity_matrix(phrases):
    n = len(phrases)
    distance_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                rouge_scorer_instance = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
                rouge_scores = rouge_scorer_instance.score(phrases[i], phrases[j])
                rouge_score = rouge_scores['rougeL'].fmeasure
                distance_matrix[i,j] = rouge_score
            else:
                distance_matrix[i, j] = 0.0  
    return distance_matrix

sentences = extract_sentences_by_creator()
bart_1 = sentences['bart_1']
bart_2 = sentences['bart_2']
t5_1 = sentences['t5_1']
t5_2 = sentences['t5_2']
FST_sentence = sentences['FST'] 

# Calcul de la longueur moyenne des phrases pour chaque créateur

def average_sentence_length(sentences):
    return sum(len(sentence.split()) for sentence in sentences) / len(sentences)

average_sentence_length_bart_1 = average_sentence_length(bart_1)
average_sentence_length_bart_2 = average_sentence_length(bart_2)
average_sentence_length_t5_1 = average_sentence_length(t5_1)
average_sentence_length_t5_2 = average_sentence_length(t5_2)
average_sentence_length_FST = average_sentence_length(FST_sentence)

print("Average length of sentences in BART_1 : " + str(average_sentence_length_bart_1))
print("Average length of sentences in BART_2 : " + str(average_sentence_length_bart_2))
print("Average length of sentences in T5_1 : " + str(average_sentence_length_t5_1))
print("Average length of sentences in T5_2 : " + str(average_sentence_length_t5_2))
print("Average length of sentences in FST : " + str(average_sentence_length_FST))


def plot_diversity_matrix(matrix, title, filename):
    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Score de diversité')
    plt.xlabel('Index de la phrase')
    plt.ylabel('Index de la phrase')
    plt.title(title)
    plt.savefig(filename)
    plt.close()

def calculer_score_rouge_moyen(matrix):
    moyenne = 0
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if i != j:
                moyenne += matrix[i][j]
    return moyenne / (len(matrix) * (len(matrix) - 1))

def calculer_ecart_type(matrix):
    moyenne = calculer_score_rouge_moyen(matrix)
    somme = 0
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if i != j:
                somme += (matrix[i][j] - moyenne) ** 2
    return (somme / (len(matrix) * (len(matrix) - 1))) ** 0.5

# Création de la matrice de diversité pour bart_1
bart_1_selected_sentences = random.sample(bart_1, 75)
diversity_matrix_bart_1 = create_diversity_matrix(bart_1_selected_sentences) 
plot_diversity_matrix(diversity_matrix_bart_1, 'Diversité des phrases - Bart_1', 'plot/diversity_matrix_bart_1.png')
#Moyenne des scores ROUGE et écart type (sans compter les coeff diagonaux de la matrice)
print("Moyenne des scores ROUGE pour Bart_1 : " + str(calculer_score_rouge_moyen(diversity_matrix_bart_1)))


# Création de la matrice de diversité pour bart_2
bart_2_selected_sentences = random.sample(bart_2, 75)
diversity_matrix_bart_2 = create_diversity_matrix(bart_2_selected_sentences)
plot_diversity_matrix(diversity_matrix_bart_2, 'Diversité des phrases - Bart_2', 'plot/diversity_matrix_bart_2.png')
print("Moyenne des scores ROUGE pour Bart_2 : " + str(calculer_score_rouge_moyen(diversity_matrix_bart_2))) 

# Création de la matrice de diversité pour t5_1
t5_1_selected_sentences = random.sample(t5_1, 75)
diversity_matrix_t5_1 = create_diversity_matrix(t5_1_selected_sentences)
plot_diversity_matrix(diversity_matrix_t5_1, 'Diversité des phrases - T5_1', 'plot/diversity_matrix_t5_1.png')
print("Moyenne des scores ROUGE pour T5_1 : " + str(calculer_score_rouge_moyen(diversity_matrix_t5_1)))

# Création de la matrice de diversité pour t5_2
t5_2_selected_sentences = random.sample(t5_2, 75)
diversity_matrix_t5_2 = create_diversity_matrix(t5_2_selected_sentences)
plot_diversity_matrix(diversity_matrix_t5_2, 'Diversité des phrases - T5_2', 'plot/diversity_matrix_t5_2.png')
print("Moyenne des scores ROUGE pour T5_2 : " + str(calculer_score_rouge_moyen(diversity_matrix_t5_2)))

# Création de la matrice de diversité pour FST
FST_sentence = random.sample(FST_sentence, 75)
diversity_matrix_FST = create_diversity_matrix(FST_sentence)
plot_diversity_matrix(diversity_matrix_FST, 'Diversité des phrases - FST', 'plot/diversity_matrix_FST.png')
print("Moyenne des scores ROUGE pour FST : " + str(calculer_score_rouge_moyen(diversity_matrix_FST)))