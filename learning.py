import json
from collections import Counter
from scipy.stats import spearmanr
import numpy as np
import matplotlib.pyplot as plt

path = '/u/haoboxu/work/verl/conf_results/conf_results_qwen3-4b_math4_traces10.json'
dic = json.load(open(path, 'r'))
print(dic[0])
exit(0)
