import pandas as pd
import matplotlib.pyplot as plt
import data_parser

data_bart_1, data_bart_2, data_t5_1, data_t5_2, data_FST, combined_data = data_parser.parser()

# Sort the data
combined_data = {k: v for k, v in sorted(combined_data.items(), key=lambda item: item[1], reverse=True)}

# Calculate the proportion of each creator for each word
proportions = {}
for word in combined_data:
    proportions[word] = [data_bart_1.get(word, 0)/combined_data.get(word, 0),
                         data_bart_2.get(word, 0)/combined_data.get(word, 0),
                         data_t5_1.get(word, 0)/combined_data.get(word, 0),
                         data_t5_2.get(word, 0)/combined_data.get(word, 0),
                         data_FST.get(word, 0)/combined_data.get(word, 0)]

# Create subplots for each creator
fig, axs = plt.subplots(5, 1, figsize=(15, 15))

# Define colors for each creator
colors = ['blue', 'green', 'red', 'orange', 'purple']

# Iterate over each creator and plot the proportion for each word
for i, creator in enumerate(['data_bart_1', 'data_bart_2', 'data_t5_1', 'data_t5_2', 'data_FST']):
    creator_proportions = [proportions[word][i] for word in combined_data.keys()]
    
    # Calculer la moyenne glissante
    rolling_mean = pd.Series(creator_proportions).rolling(window=30).mean()
    
    axs[i].plot(range(len(combined_data)), creator_proportions, color=colors[i], label='Original Data')
    axs[i].plot(range(len(combined_data)), rolling_mean, color='black', linestyle='dashed', label='Rolling Mean')
    
    axs[i].set_xlabel('Words')
    axs[i].set_ylabel('Proportion')
    axs[i].set_title(f'Proportion of {creator} in Each Word with Rolling Mean')
    axs[i].set_xticks([])  # Supprimer les marques d'axe
    axs[i].legend()

plt.tight_layout()
plt.show()
