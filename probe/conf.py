import json
from tqdm import trange
import random
import pickle

# data = json.load(open('/u/haoboxu/work/verl/conf_results_qwen3-4b_math5_traces10_.json', 'r'))
# print(len(data))
# data = [d for d in data if len(d['correctness']) > 0]
# print([d['correctness'] for d in data])
# cor = [d['correctness'][-1] for d in data]
# cof = [d['min_confs']['128'] for d in data]
# print(cor[0], cof[0])
# cor, cof = [], []
# for d in data:
#     if len(d['correctness']) == 0:
#         continue
#     cor.append(d['correctness'][-1])
#     cof.append(d['min_confs'])


from scipy.stats import spearmanr


# pickle.dump([cor, cof], open('math5_deepconf.pkl', 'wb'))
# exit(0)
cor, coff = pickle.load(open('dapo17k_deepconf.pkl', 'rb'))
for j in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
    cof = [d[str(j)] for d in coff]
    pairs = [(a, b) for (a, b) in zip(cor, cof)]
    random.shuffle(pairs)
    pairs = list(zip(cor, cof))
    print(spearmanr(cor, cof))
    # pos = [p for p in pairs if int(p[0]) == 1]
    # neg = [p for p in pairs if int(p[0]) == 0]

    # k = min(len(pos), len(neg))
    # pairs = pos[:k] + neg[:k]   # 50/50
    # cor = [p[0] for p in pairs]
    # cof = [p[1] for p in pairs]
    # print(cor, cof)
    best_ans = 0
    best_thres = None
    for i in trange(0, 2500 * 5):
        thres = float(i) / 100 / 5
        # print(cof[0], cor[0])
        ans = [int(c > thres) == int(_) for c, _ in zip(cof, cor)]
        # print(ans)
        if sum(ans) / len(ans) > best_ans:
            best_ans = sum(ans) / len(ans)
            best_thres = thres
    print(j, best_ans, best_thres)