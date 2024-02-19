import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import data_parser

data_bart_1, data_bart_2, data_t5_1, data_t5_2, data_FST, combined_data = data_parser.parser(delete_useless_words=True)

#Sort the data
data_bart_1 = {k: v for k, v in sorted(data_bart_1.items(), key=lambda item: item[1], reverse=True)}
data_bart_2 = {k: v for k, v in sorted(data_bart_2.items(), key=lambda item: item[1], reverse=True)}
data_t5_1 = {k: v for k, v in sorted(data_t5_1.items(), key=lambda item: item[1], reverse=True)}
data_t5_2 = {k: v for k, v in sorted(data_t5_2.items(), key=lambda item: item[1], reverse=True)}
data_FST = {k: v for k, v in sorted(data_FST.items(), key=lambda item: item[1], reverse=True)}
combined_data = {k: v for k, v in sorted(combined_data.items(), key=lambda item: item[1], reverse=True)}

#Print number of words
print("Number of words in BART_1: ", len(data_bart_1))
print("Number of words in BART_2: ", len(data_bart_2))
print("Number of words in T5_1: ", len(data_t5_1))
print("Number of words in T5_2: ", len(data_t5_2))
print("Number of words in FST: ", len(data_FST))
print("Number of words in total: ", len(combined_data))

#Find the proportion of each creator in the most used words
n = 50
most_used_words = list(combined_data.keys())[:n]
proportions = {}
for word in most_used_words:
    proportions[word] = [data_bart_1.get(word, 0), data_bart_2.get(word, 0), data_t5_1.get(word, 0), data_t5_2.get(word, 0), data_FST.get(word, 0)]

# Convert proportions dictionary to a DataFrame
proportions_df = pd.DataFrame(proportions, index=['BART_1', 'BART_2', 'T5_1', 'T5_2', 'FST'])

# Transpose the DataFrame
proportions_df = proportions_df.T

# Plot the stacked bar chart
proportions_df.plot(kind='bar', stacked=True)

# Set the title and labels
plt.title('Stacked Bar Chart - Most Used Words')
plt.xlabel('Words')
plt.ylabel('Proportions')
plt.xticks(rotation=45)

# Show the plot
plt.show()
