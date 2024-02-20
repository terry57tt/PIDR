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


def evaluate_metrics(reference, hypothesis):
    reference_tokens = tokenize_phrase(reference)
    hypothesis_tokens = tokenize_phrase(hypothesis)

    reference_bigrams = [' '.join(reference_tokens[i:i+4]) for i in range(len(reference_tokens)-1)]
    hypothesis_bigrams = [' '.join(hypothesis_tokens[i:i+4]) for i in range(len(hypothesis_tokens)-1)]

    reference_bigrams_str = ' '.join(reference_bigrams)
    hypothesis_bigrams_str = ' '.join(hypothesis_bigrams)

    rouge_scorer_instance = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = rouge_scorer_instance.score(reference_bigrams_str, hypothesis_bigrams_str)
    rouge_score = rouge_scores['rougeL'].fmeasure

    return rouge_score


def evaluate_phrases(createur, idInterpretation_id):
    with open('data/captions.csv', 'r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            if row['createur'] == createur and row['idInterpretation_id'] == str(idInterpretation_id):
                return row['caption']


def create_matrix(interpretationId):
    rouge = np.zeros((5, 5))

    creators = ['bart_1', 'bart_2', 't5_1', 't5_2', 'FST']

    for i in range(5):
        for j in range(5):
            reference = evaluate_phrases(creators[i], interpretationId)
            hypothesis = evaluate_phrases(creators[j], interpretationId)
            rouge[i, j] = evaluate_metrics(reference, hypothesis)

    return rouge


def plot_matrix(matrix, title, filename):
    plt.imshow(matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar(label=title)
    plt.xticks(np.arange(5), ['bart_1', 'bart_2', 't5_1', 't5_2', 'FST'])
    plt.yticks(np.arange(5), ['bart_1', 'bart_2', 't5_1', 't5_2', 'FST'])
    plt.xlabel('Hypothèses')
    plt.ylabel('Références')
    plt.title(title)
    plt.savefig(filename)
    plt.close()

def get_random_interpretation_id():
    interpretation_ids = []
    with open('data/captions.csv', 'r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            interpretation_ids.append(row['idInterpretation_id'])
    return random.choice(interpretation_ids)

def means_and_std(nb_test):
    scores_rouge = np.zeros((5, 5, nb_test))

    for test in range(nb_test):

        interpretation_id = get_random_interpretation_id()
        print(interpretation_id)
        rouge = create_matrix(interpretation_id)
        scores_rouge[:, :, test] = rouge

    mean_rouge = np.mean(scores_rouge, axis=2)
    std_rouge = np.std(scores_rouge, axis=2)

    return (mean_rouge, std_rouge)

rouge = means_and_std(1000)

plot_matrix(rouge[0], 'ROUGE Score (mean)', 'plot/n_grammes/rouge_score_mean_4.png')
plot_matrix(rouge[1], 'ROUGE Score (std)', 'plot/n_grammes/rouge_score_std_4.png')


