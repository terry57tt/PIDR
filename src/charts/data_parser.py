import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def parser(delete_useless_words=False):

    file_path = 'data/captions.csv'

    data = pd.read_csv(file_path, header=0)

    datasets = {}

    for createur, data in data.groupby('createur'):
        datasets[createur] = data

    data_bart_1 = {}
    data_bart_2 = {}
    data_t5_1 = {}
    data_t5_2 = {}
    data_FST = {}
    combined_data = {}

    useless_words = ["the","i","a","it","not","way","am","are","is","what","me","this","because","an","and","or","but","as","of","at","by","for","with","about","to","from","in","on","off","i'm","they","that","he","she","we","you","your","my","his","her","their","our","its","it's","there","here","where","when","how","why","who"]
    
    for createur in datasets:
        for _, data_row in datasets[createur].iterrows():
            string = data_row['caption']
            string = string.lower()
            tab = string.split(' ')

            for i in range(len(tab)):
                str = tab[i]
                if "#" in str:
                    tab.pop(i)
                    tab.extend(str.split('#'))
            
            for i in range(len(tab)):
                str = tab[i]
                if "&" in str:
                    tab.pop(i)
                    tab.extend(str.split('&'))

            if delete_useless_words:
                ntab = []
                for word in tab:
                    if word not in useless_words:
                        ntab.append(word)
                tab = ntab                        

            for word in tab:
                if word in combined_data:
                    combined_data[word] += 1
                else:
                    combined_data[word] = 1

            if createur == 'bart_1':
                for word in tab:
                    if word in data_bart_1:
                        data_bart_1[word] += 1
                    else:
                        data_bart_1[word] = 1
            elif createur == 'bart_2':
                for word in tab:
                    if word in data_bart_2:
                        data_bart_2[word] += 1
                    else:
                        data_bart_2[word] = 1
            elif createur == 't5_1':
                for word in tab:
                    if word in data_t5_1:
                        data_t5_1[word] += 1
                    else:
                        data_t5_1[word] = 1
            elif createur == 't5_2':
                for word in tab:
                    if word in data_t5_2:
                        data_t5_2[word] += 1
                    else:
                        data_t5_2[word] = 1
            elif createur == 'FST':
                for word in tab:
                    if word in data_FST:
                        data_FST[word] += 1
                    else:
                        data_FST[word] = 1
            else:
                pass
    return data_bart_1, data_bart_2, data_t5_1, data_t5_2, data_FST, combined_data

data = parser()