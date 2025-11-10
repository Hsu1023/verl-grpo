import json
from collections import Counter
from scipy.stats import spearmanr
import numpy as np
import matplotlib.pyplot as plt

path = '/u/haoboxu/work/verl/conf_results_math1.json'
dic = json.load(open(path, 'r'))
# print(len([i for i in dic if len(i['correctness']) > 2]))

y = []
for length in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
    correctness_list = [i['correctness'][-1] for i in dic]
    conf_list = [i['min_confs'][str(length)] for i in dic]
    y.append(spearmanr(correctness_list, conf_list).correlation)
    print(f'length: {length}, spearmanr: {spearmanr(correctness_list, conf_list)}')
    
plt.plot(np.arange(len(y)), y)
plt.ylabel('Spearmanr')
plt.xlabel('Log Window Length')
plt.xticks(np.arange(len(y)), [str(l) for l in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]])
plt.ylim(-0.1, 0.6)
plt.title('Spearmanr: correctness vs. Log Window Length (min_conf)')
plt.savefig('dapo_conf_correctness_spearmanr_min_conf.png')
plt.clf()

y = []
for length in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
    correctness_list = [i['correctness'][-1] for i in dic]
    conf_list = [i['last_confs'][str(length)] for i in dic]
    y.append(spearmanr(correctness_list, conf_list).correlation)
    print(f'length: {length}, spearmanr: {spearmanr(correctness_list, conf_list)}')
    
plt.plot(np.arange(len(y)), y)
plt.ylabel('Spearmanr')
plt.xlabel('Log Window Length')
plt.xticks(np.arange(len(y)), [str(l) for l in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]])
plt.ylim(-0.1, 0.6)
plt.title('Spearmanr: correctness vs. Log Window Length (last_conf)')
plt.savefig('dapo_conf_correctness_spearmanr_last_conf.png')
plt.clf()



# alignment
correctness_list = [i['correctness'][-1] for i in dic if len(i['correctness']) > 2]
conf_list = [i['correctness'][0] for i in dic if len(i['correctness']) > 2]
print('alignment between first and last box:', spearmanr(correctness_list, conf_list))

# token length
correctness_list = [i['correctness'][-1] for i in dic]
conf_list = [i['length_of_confs'] for i in dic]
print('token length', spearmanr(correctness_list, conf_list))

# mean boxed confs
conf_list = [i['mean_boxed_confs'][-1] for i in dic]
print('mean boxed confs', spearmanr(correctness_list, conf_list))


# mean boxed confs
conf_list = [sum(i['boxed_confs'][0])/len(i['boxed_confs'][0]) for i in dic]
print('First boxed confs', spearmanr(correctness_list, conf_list))

# mean boxed confs
conf_list = [sum(i['boxed_confs'][-1])/len(i['boxed_confs'][-1]) for i in dic]
print('Last boxed confs', spearmanr(correctness_list, conf_list))


# span
conf_list = [i['spans'][0][1] for i in dic]
print('first span', spearmanr(correctness_list, conf_list))


counter = Counter()
for c in correctness_list:
    counter[c] += 1
print(counter)