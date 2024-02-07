import pandas as pd
import matplotlib.pyplot as plt

file_path = 'src/data.csv'

data = pd.read_csv(file_path, header=0)

datasets = {}

for createur, data in data.groupby('createur'):
    datasets[createur] = data

data_bart_1 = {}
data_bart_2 = {}
data_t5_1 = {}
data_t5_2 = {}
data_FST = {}

for createur in datasets:
    for _, data_row in datasets[createur].iterrows():
        string = data_row['caption']
        tab = string.split(' ')

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

#Sort the data
data_bart_1 = {k: v for k, v in sorted(data_bart_1.items(), key=lambda item: item[1], reverse=True)}
data_bart_2 = {k: v for k, v in sorted(data_bart_2.items(), key=lambda item: item[1], reverse=True)}
data_t5_1 = {k: v for k, v in sorted(data_t5_1.items(), key=lambda item: item[1], reverse=True)}
data_t5_2 = {k: v for k, v in sorted(data_t5_2.items(), key=lambda item: item[1], reverse=True)}
data_FST = {k: v for k, v in sorted(data_FST.items(), key=lambda item: item[1], reverse=True)}

#Print number of words
print("Number of words in BART_1: ", len(data_bart_1))
print("Number of words in BART_2: ", len(data_bart_2))
print("Number of words in T5_1: ", len(data_t5_1))
print("Number of words in T5_2: ", len(data_t5_2))
print("Number of words in FST: ", len(data_FST))

#Print the 10 most used words        
fig, axs = plt.subplots(5, 1, figsize=(10, 15), sharex=False, sharey=True)

# Plot for createur 'bart_1'
axs[0].bar(list(data_bart_1.keys())[:10], list(data_bart_1.values())[:10])
axs[0].set_title('BART_1')
axs[0].set_ylabel('Frequency')

# Plot for createur 'bart_2'
axs[1].bar(list(data_bart_2.keys())[:10], list(data_bart_2.values())[:10])
axs[1].set_title('BART_2')
axs[1].set_ylabel('Frequency')

# Plot for createur 't5_1'
axs[2].bar(list(data_t5_1.keys())[:10], list(data_t5_1.values())[:10])
axs[2].set_title('T5_1')
axs[2].set_ylabel('Frequency')

# Plot for createur 't5_2'
axs[3].bar(list(data_t5_2.keys())[:10], list(data_t5_2.values())[:10])
axs[3].set_title('T5_2')
axs[3].set_ylabel('Frequency')

# Plot for createur 'FST'
axs[4].bar(list(data_FST.keys())[:10], list(data_FST.values())[:10])
axs[4].set_title('FST')
axs[4].set_ylabel('Frequency')

plt.suptitle('Top 10 Most Used Words by Each IA')
plt.tight_layout()
plt.show()