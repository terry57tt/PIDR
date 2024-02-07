import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import data_parser

data_bart_1, data_bart_2, data_t5_1, data_t5_2, data_FST, combined_data = data_parser.parser()

# Sort the data
data_bart_1 = {k: v for k, v in sorted(data_bart_1.items(), key=lambda item: item[1], reverse=False)}
data_bart_2 = {k: v for k, v in sorted(data_bart_2.items(), key=lambda item: item[1], reverse=False)}
data_t5_1 = {k: v for k, v in sorted(data_t5_1.items(), key=lambda item: item[1], reverse=False)}
data_t5_2 = {k: v for k, v in sorted(data_t5_2.items(), key=lambda item: item[1], reverse=False)}
data_FST = {k: v for k, v in sorted(data_FST.items(), key=lambda item: item[1], reverse=False)}
combined_data = {k: v for k, v in sorted(combined_data.items(), key=lambda item: item[1], reverse=False)}

# Find the proportion of each creator in the least used words
least_used_words = [word for word, count in combined_data.items() if count <= 10]
proportions = {}
for word in least_used_words:
    proportions[word] = [data_bart_1.get(word, 0), data_bart_2.get(word, 0), data_t5_1.get(word, 0), data_t5_2.get(word, 0), data_FST.get(word, 0)]

#Sum the proportions
sums = [0, 0, 0, 0, 0]
for word in least_used_words:
    for i in range(5):
        sums[i] += proportions[word][i]

# Calculate the percentages
percentages = [sum_value / sum(sums) * 100 for sum_value in sums]

# Labels for the pie chart
labels = ['Bart 1', 'Bart 2', 'T5 1', 'T5 2', 'FST']

# Create the pie chart
plt.pie(percentages, labels=labels, autopct='%1.1f%%')

# Add a title
plt.title('Proportion of which AI created the least used words')

# Display the chart
plt.show()