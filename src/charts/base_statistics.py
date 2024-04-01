import pandas as pd
import matplotlib.pyplot as plt
import data_parser
import plotly.express as px

file_path = 'data/captions.csv'

raw_data = pd.read_csv(file_path, header=0)

#### Data statistics
# Number of words by creator
number_of_words_by_creator = raw_data.groupby('createur')['caption'].apply(lambda x: x.str.split().str.len().sum())
print(f'Number of raw words by creator: {number_of_words_by_creator}')

# Box plot of number of words by sentence by creator
raw_data['number_of_words'] = raw_data['caption'].str.split().str.len()
px.box(raw_data, x='createur', y='number_of_words', title='Number of words by sentence by creator').show()

#### Sample of mean sentences of each creator compared to outliers
# Mean number of words for FST
mean_FST = raw_data.loc[raw_data['createur'] == 'FST', 'number_of_words'].mean()
mean_sentences_FST = raw_data[(raw_data['createur'] == 'FST') & (raw_data['number_of_words'] == round(mean_FST))][:5]
outliers_FST = raw_data[(raw_data['createur'] == 'FST') & (raw_data['number_of_words'] < 5)]
print(f'Mean number of words for FST: {mean_FST}')
print(f'Mean sentences for FST:')
print(mean_sentences_FST['caption'])
print(f'Outliers for FST:')
print(outliers_FST['caption'])

# Mean number of words for t5_1
mean_t5_1 = raw_data.loc[raw_data['createur'] == 't5_1', 'number_of_words'].mean()
mean_sentences_t5_1 = raw_data[(raw_data['createur'] == 't5_1') & (raw_data['number_of_words'] == round(mean_t5_1))][:5]
outliers_t5_1 = raw_data[(raw_data['createur'] == 't5_1') & ((raw_data['number_of_words'] > 20) | (raw_data['number_of_words'] < 9))]
print(f'Mean number of words for t5_1: {mean_t5_1}')
print(f'Mean sentences for t5_1:')
print(mean_sentences_t5_1['caption'])
print(f'Outliers for t5_1:')
print(outliers_t5_1['caption'])


# Mean number of words for t5_2
mean_t5_2 = raw_data.loc[raw_data['createur'] == 't5_2', 'number_of_words'].mean()
mean_sentences_t5_2 = raw_data[(raw_data['createur'] == 't5_2') & (raw_data['number_of_words'] == round(mean_t5_2))][:5]
outliers_t5_2 = raw_data[(raw_data['createur'] == 't5_2') & ((raw_data['number_of_words'] > 15) | (raw_data['number_of_words'] < 7))]
print(f'Mean number of words for t5_2: {mean_t5_2}')
print(f'Mean sentences for t5_2:')
print(mean_sentences_t5_2['caption'])
print(f'Outliers for t5_2:')
print(outliers_t5_2['caption'])

# Mean number of words for bart_1
mean_bart_1 = raw_data.loc[raw_data['createur'] == 'bart_1', 'number_of_words'].mean()
mean_sentences_bart_1 = raw_data[(raw_data['createur'] == 'bart_1') & (raw_data['number_of_words'] == round(mean_bart_1))][:5]
outliers_bart_1 = raw_data[(raw_data['createur'] == 'bart_1') & (raw_data['number_of_words'] > 21)]
print(f'Mean number of words for bart_1: {mean_bart_1}')
print(f'Mean sentences for bart_1:')
print(mean_sentences_bart_1['caption'])
print(f'Outliers for bart_1:')
print(outliers_bart_1['caption'])

# Mean number of words for bart_2
mean_bart_2 = raw_data.loc[raw_data['createur'] == 'bart_2', 'number_of_words'].mean()
mean_sentences_bart_2 = raw_data[(raw_data['createur'] == 'bart_2') & (raw_data['number_of_words'] == round(mean_bart_2))][:5]
outliers_bart_2 = raw_data[(raw_data['createur'] == 'bart_2') & (raw_data['number_of_words'] > 25)]
print(f'Mean number of words for bart_2: {mean_bart_2}')
print(f'Mean sentences for bart_2:')
print(mean_sentences_bart_2['caption'])
print(f'Outliers for bart_2:')
print(outliers_bart_2['caption'])







