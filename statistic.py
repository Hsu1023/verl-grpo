import json
from collections import Counter
from scipy.stats import spearmanr
import numpy as np

path = '/u/haoboxu/work/verl/conf_results_dapo.json'
dic = json.load(open(path, 'r'))
# print(len([i for i in dic if len(i['correctness']) > 2]))
correctness_list = [i['correctness'][-1] for i in dic]
# correctness_list = [i['correctness'][-1] for i in dic if len(i['correctness']) > 2]
# conf_list = [i['correctness'][0] for i in dic if len(i['correctness']) > 2]
conf_list = [i['min_confs']['1024'] for i in dic]
# conf_list = [i['length_of_confs'] for i in dic]
# print(sum(conf_list)/len(conf_list))
# conf_list = [i['mean_boxed_confs'][-1] for i in dic]
# conf_list = [i['last_confs']['1024'] for i in dic]
print(spearmanr(correctness_list, conf_list))
counter = Counter()
for c in correctness_list:
    counter[c] += 1
print(counter)