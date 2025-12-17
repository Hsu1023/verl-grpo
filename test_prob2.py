probe_min = 0.38
probe_max = 0.86
import random
import math
import random
from typing import Tuple

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def _logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1.0 - eps)
    return math.log(p / (1.0 - p))

def _calibrate_bucket_probs(a: float, b: float, pi: float) -> Tuple[float, float, float]:
    """
    Estimate p_L, p_M, p_H using only quantiles a,b and global positive rate pi.
    We approximate bucket means of r, then apply a logit-shift calibration so that
    the weighted average matches pi.
    Buckets: L: r<a (25%), M: [a,b] (50%), H: r>b (25%)
    """
    # representative r means from quantiles (minimal assumption)
    mu_L = 0.5 * a
    mu_M = 0.5 * (a + b)
    mu_H = 0.5 * (1.0 + b)

    # Solve beta in g(r)=sigmoid(logit(r)+beta) s.t. 0.25 g(mu_L)+0.5 g(mu_M)+0.25 g(mu_H)=pi
    def f(beta: float) -> float:
        gL = _sigmoid(_logit(mu_L) + beta)
        gM = _sigmoid(_logit(mu_M) + beta)
        gH = _sigmoid(_logit(mu_H) + beta)
        return 0.25 * gL + 0.5 * gM + 0.25 * gH

    # binary search beta
    lo, hi = -20.0, 20.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if f(mid) < pi:
            lo = mid
        else:
            hi = mid
    beta = 0.5 * (lo + hi)

    pL = _sigmoid(_logit(mu_L) + beta)
    pM = _sigmoid(_logit(mu_M) + beta)
    pH = _sigmoid(_logit(mu_H) + beta)
    return pL, pM, pH

def _greedy_extreme_solution(w, p, K, maximize_pos: bool):
    """
    Construct the extreme (fractional) solution that maximizes/minimizes kept positive mass
    under keep budget K, by filling buckets in order of p.
    Returns (kL,kM,kH, kept_pos_mass).
    """
    idx = list(range(3))
    idx.sort(key=lambda i: p[i], reverse=maximize_pos)

    keep_mass = [0.0, 0.0, 0.0]  # absolute kept mass in each bucket (<= w[i])
    remaining = K
    for i in idx:
        take = min(w[i], remaining)
        keep_mass[i] = take
        remaining -= take
        if remaining <= 1e-12:
            break

    k = [keep_mass[i] / w[i] if w[i] > 0 else 0.0 for i in range(3)]
    kept_pos_mass = sum(keep_mass[i] * p[i] for i in range(3))
    return k[0], k[1], k[2], kept_pos_mass

def _solve_three_bucket_keep_probs(a: float, b: float, pi: float, K: float = 0.5, q: float = 0.5):
    """
    Solve for keep probabilities (kL,kM,kH) for the 3 buckets.
    Tries to match:
      sum w_i k_i = K
      sum w_i p_i k_i = K*q
    with a monotone preference direction:
      if pi < q => prefer keeping high bucket (kL <= kM <= kH)
      if pi > q => prefer keeping low bucket  (kL >= kM >= kH)
    If exact match is infeasible, returns the extreme solution that gets closest.
    """
    assert 0.0 < a < b < 1.0, "Expect 0<a<b<1 from quantiles"
    assert 0.0 < pi < 1.0
    assert 0.0 < K <= 1.0
    assert 0.0 < q < 1.0

    wL, wM, wH = 0.25, 0.5, 0.25
    w = [wL, wM, wH]

    pL, pM, pH = _calibrate_bucket_probs(a, b, pi)
    p = [pL, pM, pH]

    target_pos_mass = K * q

    # feasibility range via greedy extremes
    kL_max, kM_max, kH_max, pos_max = _greedy_extreme_solution(w, p, K, maximize_pos=True)
    kL_min, kM_min, kH_min, pos_min = _greedy_extreme_solution(w, p, K, maximize_pos=False)

    if target_pos_mass > pos_max + 1e-10:
        # cannot reach q; return best (max pos)
        return (kL_max, kM_max, kH_max), (pL, pM, pH), {"feasible": False, "mode": "max_pos", "pos_mass": pos_max}
    if target_pos_mass < pos_min - 1e-10:
        # cannot reach q; return best (min pos)
        return (kL_min, kM_min, kH_min), (pL, pM, pH), {"feasible": False, "mode": "min_pos", "pos_mass": pos_min}

    # exact feasible: enumerate boundary cases (fix one var to 0/1, solve 2x2)
    candidates = []
    # direction preference: enrich positives if pi<q, else enrich negatives if pi>q
    enrich_pos = (pi < q - 1e-12)
    enrich_neg = (pi > q + 1e-12)

    def add_candidate(kL, kM, kH):
        # bounds
        if not (0.0 <= kL <= 1.0 and 0.0 <= kM <= 1.0 and 0.0 <= kH <= 1.0):
            return
        # keep rate
        if abs(wL*kL + wM*kM + wH*kH - K) > 1e-6:
            return
        # pos mass
        if abs(wL*pL*kL + wM*pM*kM + wH*pH*kH - target_pos_mass) > 1e-5:
            return
        # monotone preference (only enforce if pi not ~ q)
        if enrich_pos and not (kL <= kM <= kH + 1e-12):
            return
        if enrich_neg and not (kL >= kM >= kH - 1e-12):
            return
        candidates.append((kL, kM, kH))

    # solve with one fixed var
    for fixed_idx in [0, 1, 2]:
        for fixed_val in [0.0, 1.0]:
            # unknown indices
            unk = [i for i in [0,1,2] if i != fixed_idx]
            i, j = unk[0], unk[1]

            # equations:
            # w_i k_i + w_j k_j = K - w_f k_f
            # w_i p_i k_i + w_j p_j k_j = target - w_f p_f k_f
            rhs1 = K - w[fixed_idx]*fixed_val
            rhs2 = target_pos_mass - w[fixed_idx]*p[fixed_idx]*fixed_val

            # solve 2x2:
            # [w_i,       w_j      ] [k_i] = [rhs1]
            # [w_i p_i,   w_j p_j  ] [k_j]   [rhs2]
            det = w[i]*w[j]*p[j] - w[j]*w[i]*p[i]  # = w_i w_j (p_j - p_i)
            if abs(det) < 1e-12:
                continue

            k_i = ( rhs1*(w[j]*p[j]) - rhs2*w[j] ) / det
            k_j = ( rhs2*w[i] - rhs1*(w[i]*p[i]) ) / det

            k = [None, None, None]
            k[fixed_idx] = fixed_val
            k[i] = k_i
            k[j] = k_j
            add_candidate(k[0], k[1], k[2])

    # pick the "most extreme" in desired direction
    if candidates:
        if enrich_pos:
            # prefer higher kH, then kM, then kL
            candidates.sort(key=lambda t: (t[2], t[1], t[0]), reverse=True)
        elif enrich_neg:
            # prefer higher kL, then kM, then kH
            candidates.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
        else:
            # pi ~= q: any candidate; pick the most uniform (min variance)
            def var(t):
                m = sum(t)/3.0
                return sum((x-m)**2 for x in t)
            candidates.sort(key=var)
        best = candidates[0]
        return best, (pL, pM, pH), {"feasible": True, "mode": "exact"}

    # Shouldn't happen if feasible range check passed, but just in case: fall back to closest extreme
    # choose extreme closer to target_pos_mass
    if abs(pos_max - target_pos_mass) < abs(pos_min - target_pos_mass):
        return (kL_max, kM_max, kH_max), (pL, pM, pH), {"feasible": False, "mode": "fallback_max_pos", "pos_mass": pos_max}
    else:
        return (kL_min, kM_min, kH_min), (pL, pM, pH), {"feasible": False, "mode": "fallback_min_pos", "pos_mass": pos_min}

# ------------------- integrate into your stopping -------------------

class ProbeEarlyStopper:
    def __init__(self, probe_min: float, probe_max: float, pos_label_ratio: float,
                 keep_rate: float = 0.5, target_pos_in_kept: float = 0.5, seed: int = 0):
        self.probe_min = probe_min
        self.probe_max = probe_max
        self.pos_label_ratio = pos_label_ratio
        self.keep_rate = keep_rate
        self.target_pos_in_kept = target_pos_in_kept
        # random.seed(seed)

        (self.kL, self.kM, self.kH), (self.pL, self.pM, self.pH), self.meta = _solve_three_bucket_keep_probs(
            probe_min, probe_max, pos_label_ratio, K=keep_rate, q=target_pos_in_kept
        )

    def should_stop(self, r: float) -> bool:
        """True means stop generation."""
        a, b = self.probe_min, self.probe_max
        if r < a:
            keep_prob = self.kL
        elif r <= b:
            keep_prob = self.kM
        else:
            keep_prob = self.kH
        keep = (random.random() < keep_prob)
        return (not keep)
probe_min=0.38
probe_max=0.80
stop = ProbeEarlyStopper(probe_min, probe_max, 0.5, keep_rate=0.5, target_pos_in_kept=0.5)
        
sum = []
for i in range(10000):
    r = random.random()
    # if distribution(None, r):
    if stop.should_stop(r):
        sum.append(1)
    else:
        sum.append(0)
print(sum.count(1)/len(sum))