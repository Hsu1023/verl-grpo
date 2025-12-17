probe_min = 0.0
probe_max = 0.7
import random
# def distribution(self, r, m=0.5):
#     assert 0.0 <= r <= 1.0
    
#     m = 1 - m

#     a = probe_min   # e.g. 0.12
#     b = probe_max   # e.g. 0.36

#     # 如果区间非法，退化为 p=0.5
#     if a < 0 or b < 0 or a >= b:
#         return (random.random() < m)

#     B = b - a

#     # 理论上要求 B <= 0.75，否则下面的 p_out 可能为负
#     # 对你 25%-75% 这种用法，通常 B 会 < 0.75
#     if B > 0.75:
#         B = 0.75  # 或者直接 fallback

#     # K = ∫ bump(r) dr
#     K = (2.0 / 3.0) * B

#     # 期望约束: 0.5 = p_out + (1 - p_out)*K
#     # => p_out = (0.5 - K) / (1 - K)
#     p_out = (m - K) / (1.0 - K)

#     # numerical safety
#     if p_out < 0.0:
#         p_out = 0.0
#     elif p_out > 1.0:
#         p_out = 1.0

#     # 计算 p(r)
#     if r < a or r > b:
#         p = p_out
#     else:
#         u = (r - a) / B    # in [0,1]
#         bump = 4.0 * u * (1.0 - u)   # in [0,1]
#         p = p_out + (1.0 - p_out) * bump

#     # p 肯定在 [p_out, 1] ⊆ [0,1]，不需要 clip 了

#     # 注意：如果你想 "区间内 True 概率大"
#     # ret=True 的概率就应该是 p，而不是 1-p
#     ret = (random.random() < p)

#     return ret


def distribution(self, r, m=0.2):
            
    a = probe_min   # e.g. 0.12
    b = probe_max   # e.g. 0.36
    # -1是不进行early_exit
    if a < 0 or b < 0:
        return False
    
    import random
    assert 0.0 <= r <= 1.0
    m = 1-m


    
    # 如果区间非法，退化为 p=0.5
    if a >= b:
        return (random.random() < m)

    B = b - a

    # 理论上要求 B <= 0.75，否则下面的 p_out 可能为负
    # 对你 25%-75% 这种用法，通常 B 会 < 0.75
    if B > 0.75:
        B = 0.75  # 或者直接 fallback

    # K = ∫ bump(r) dr
    K = (2.0 / 3.0) * B

    # 期望约束: 0.5 = p_out + (1 - p_out)*K
    # => p_out = (0.5 - K) / (1 - K)
    p_out = (m - K) / (1.0 - K)

    # numerical safety
    if p_out < 0.0:
        p_out = 0.0
    elif p_out > 1.0:
        p_out = 1.0

    # 计算 p(r)
    if r < a or r > b:
        p = p_out
    else:
        u = (r - a) / B    # in [0,1]
        bump = 4.0 * u * (1.0 - u)   # in [0,1]
        p = p_out + (1.0 - p_out) * bump

    # p 肯定在 [p_out, 1] ⊆ [0,1]，不需要 clip 了

    # 注意：如果你想 "区间内 True 概率大"
    # ret=True 的概率就应该是 p，而不是 1-p
    ret = (random.random() < p)

    # return not ret
    return ret
        
sum = []
for i in range(16):
    r = random.random() * 0.2
    if distribution(None, r, m=0.5):
        sum.append(1)
    else:
        sum.append(0)
print(sum.count(1)/len(sum))