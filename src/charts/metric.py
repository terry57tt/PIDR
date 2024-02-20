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

    # BLEU score
    smoothing_function = SmoothingFunction().method4
    bleu_score = sentence_bleu([reference_tokens], hypothesis_tokens, smoothing_function=smoothing_function)

    # ROUGE score
    rouge_scorer_instance = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = rouge_scorer_instance.score(reference, hypothesis)
    rouge_score = rouge_scores['rougeL'].fmeasure

    # METEOR score
    meteor_score = single_meteor_score(reference_tokens, hypothesis_tokens)

    return bleu_score, rouge_score, meteor_score


def evaluate_phrases(createur, idInterpretation_id):
    with open('data/captions.csv', 'r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            if row['createur'] == createur and row['idInterpretation_id'] == str(idInterpretation_id):
                return row['caption']


def create_matrix(interpretationId):
    bleu = np.zeros((5, 5))
    rouge = np.zeros((5, 5))
    meteor = np.zeros((5, 5))

    creators = ['bart_1', 'bart_2', 't5_1', 't5_2', 'FST']

    for i in range(5):
        for j in range(5):
            reference = evaluate_phrases(creators[i], interpretationId)
            hypothesis = evaluate_phrases(creators[j], interpretationId)
            bleu[i, j], rouge[i, j], meteor[i, j] = evaluate_metrics(reference, hypothesis)

    return bleu, rouge, meteor


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


bleu, rouge, meteor = create_matrix(895)

plot_matrix(bleu, 'BLEU Score', 'plot/bleu_score.png')
plot_matrix(rouge, 'ROUGE Score', 'plot/rouge_score.png')
plot_matrix(meteor, 'METEOR Score', 'plot/meteor_score.png')

def get_random_interpretation_id():
    interpretation_ids = []
    with open('data/captions.csv', 'r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            interpretation_ids.append(row['idInterpretation_id'])
    return random.choice(interpretation_ids)

def means_and_std(nb_test):
    scores_bleu = np.zeros((5, 5, nb_test))
    scores_rouge = np.zeros((5, 5, nb_test))
    scores_meteor = np.zeros((5, 5, nb_test))

    for test in range(nb_test):
        if((test == 5) or (test == 100) or (test == 250) or (test == 500) or (test == 750) or (test == 1000)):
            print("loading"+str(test))
            bleu, rouge, meteor = create_matrix(interpretation_id)
            scores_bleu[:, :, test] = bleu
            scores_rouge[:, :, test] = rouge
            scores_meteor[:, :, test] = meteor

            mean_bleu = np.mean(scores_bleu, axis=2)
            std_bleu = np.std(scores_bleu, axis=2)
            mean_rouge = np.mean(scores_rouge, axis=2)
            std_rouge = np.std(scores_rouge, axis=2)
            mean_meteor = np.mean(scores_meteor, axis=2)
            std_meteor = np.std(scores_meteor, axis=2)

            bleu, rouge, meteor = (mean_bleu, std_bleu), (mean_rouge, std_rouge), (mean_meteor, std_meteor)

            plot_matrix(bleu[0], 'BLEU Score (mean)', 'plot/stat/bleu_score_mean_'+str(test)+'.png')
            plot_matrix(bleu[1], 'BLEU Score (std)', 'plot/stat/bleu_score_std_'+str(test)+'.png')

            plot_matrix(rouge[0], 'ROUGE Score (mean)', 'plot/stat/rouge_score_mean_'+str(test)+'.png')
            plot_matrix(rouge[1], 'ROUGE Score (std)', 'plot/stat/rouge_score_std_'+str(test)+'.png')

            plot_matrix(meteor[0], 'METEOR Score (mean)', 'plot/stat/meteor_score_mean_'+str(test)+'.png')
            plot_matrix(meteor[1], 'METEOR Score (std)', 'plot/stat/meteor_score_std_'+str(test)+'.png')

        interpretation_id = get_random_interpretation_id()
        print(interpretation_id)
        bleu, rouge, meteor = create_matrix(interpretation_id)
        scores_bleu[:, :, test] = bleu
        scores_rouge[:, :, test] = rouge
        scores_meteor[:, :, test] = meteor

    mean_bleu = np.mean(scores_bleu, axis=2)
    std_bleu = np.std(scores_bleu, axis=2)
    mean_rouge = np.mean(scores_rouge, axis=2)
    std_rouge = np.std(scores_rouge, axis=2)
    mean_meteor = np.mean(scores_meteor, axis=2)
    std_meteor = np.std(scores_meteor, axis=2)

    return (mean_bleu, std_bleu), (mean_rouge, std_rouge), (mean_meteor, std_meteor)

bleu, rouge, meteor = means_and_std(1010)

plot_matrix(bleu[0], 'BLEU Score (mean)', 'plot/bleu_score_mean.png')
plot_matrix(bleu[1], 'BLEU Score (std)', 'plot/bleu_score_std.png')

plot_matrix(rouge[0], 'ROUGE Score (mean)', 'plot/rouge_score_mean.png')
plot_matrix(rouge[1], 'ROUGE Score (std)', 'plot/rouge_score_std.png')

plot_matrix(meteor[0], 'METEOR Score (mean)', 'plot/meteor_score_mean.png')
plot_matrix(meteor[1], 'METEOR Score (std)', 'plot/meteor_score_std.png')


