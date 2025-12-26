from __future__ import annotations
from typing import List, Tuple, Sequence, Callable, Any, Optional
import numpy as np

def _to_np(weights: Sequence[float]) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 1:
        raise ValueError(f"weights must be 1-D, got shape {w.shape}")
    if np.any(~np.isfinite(w)):
        raise ValueError("weights contains NaN/Inf")
    return w

def _check_len(answers: Sequence[Any], weights: Sequence[float]) -> None:
    if len(answers) != len(weights):
        raise ValueError(f"len(answers)={len(answers)} != len(weights)={len(weights)}")

def _renorm_sum(w_new: np.ndarray, target_sum: float, eps: float = 1e-12) -> np.ndarray:
    s = float(w_new.sum())
    if s < eps:
        # fallback: uniform
        return np.full_like(w_new, target_sum / max(len(w_new), 1))
    return w_new * (target_sum / s)

def _renorm_prob(w_new: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    s = float(w_new.sum())
    if s < eps:
        return np.full_like(w_new, 1.0 / max(len(w_new), 1))
    return w_new / s

# -------------------------------------------------------------------
# 1) 幂次压缩：w' = (w + eps)^gamma, gamma in (0,1) 更平均
# -------------------------------------------------------------------
def weights_power_smooth(
    answers: Sequence[Any],
    weights: Sequence[float],
    gamma: float = 0.5,
    eps: float = 1e-6,
    keep_sum: bool = True,
) -> Tuple[List[Any], List[float]]:
    _check_len(answers, weights)
    w = _to_np(weights)
    if gamma <= 0:
        raise ValueError("gamma must be > 0 (use 0<gamma<1 to flatten)")
    target_sum = float(w.sum())
    w_new = np.power(np.maximum(w, 0.0) + eps, gamma)

    if keep_sum:
        w_new = _renorm_sum(w_new, target_sum)
    return list(answers), w_new.tolist()

# -------------------------------------------------------------------
# 2) 概率化 + flatten: p = w/sum(w), p' ∝ p^gamma, gamma<1 变平
#    可选混合均匀：p'' = (1-lam)p' + lam*(1/n)
# -------------------------------------------------------------------
def weights_prob_flatten(
    answers: Sequence[Any],
    weights: Sequence[float],
    gamma: float = 0.7,
    lam: float = 0.0,
    eps: float = 1e-12,
    return_as: str = "sum",  # "sum" or "prob"
) -> Tuple[List[Any], List[float]]:
    _check_len(answers, weights)
    w = _to_np(weights)
    n = len(w)
    target_sum = float(w.sum())

    p = _renorm_prob(np.maximum(w, 0.0), eps=eps)
    # flatten or sharpen
    p2 = np.power(np.maximum(p, eps), gamma)
    p2 = _renorm_prob(p2, eps=eps)

    if lam > 0:
        p2 = (1.0 - lam) * p2 + lam * (1.0 / max(n, 1))
        p2 = _renorm_prob(p2, eps=eps)

    if return_as == "prob":
        return list(answers), p2.tolist()
    elif return_as == "sum":
        return list(answers), (p2 * target_sum).tolist()
    else:
        raise ValueError("return_as must be 'sum' or 'prob'")

# -------------------------------------------------------------------
# 3) 分位数裁剪（winsorize）再归一化：适合“卡上限/长尾”
# -------------------------------------------------------------------
def weights_winsorize(
    answers: Sequence[Any],
    weights: Sequence[float],
    q_low: float = 0.01,
    q_high: float = 0.95,
    keep_sum: bool = True,
) -> Tuple[List[Any], List[float]]:
    _check_len(answers, weights)
    w = _to_np(weights)
    if not (0.0 <= q_low < q_high <= 1.0):
        raise ValueError("require 0<=q_low<q_high<=1")
    target_sum = float(w.sum())

    lo = float(np.quantile(w, q_low))
    hi = float(np.quantile(w, q_high))
    w_new = np.clip(w, lo, hi)

    if keep_sum:
        w_new = _renorm_sum(w_new, target_sum)
    return list(answers), w_new.tolist()

# -------------------------------------------------------------------
# 4) Rank / 分位数映射：最平均，但会改变原相对比例（只保序）
# -------------------------------------------------------------------
def weights_rank_map(
    answers: Sequence[Any],
    weights: Sequence[float],
    a: float = 0.2,
    b: float = 0.8,
    keep_sum: bool = True,
) -> Tuple[List[Any], List[float]]:
    _check_len(answers, weights)
    w = _to_np(weights)
    n = len(w)
    target_sum = float(w.sum())

    # ranks: smallest->0, largest->n-1
    order = np.argsort(w, kind="mergesort")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(n)

    if n <= 1:
        w_new = np.array([target_sum], dtype=np.float64)
    else:
        w_new = a + (b - a) * (ranks / (n - 1))

    if keep_sum:
        w_new = _renorm_sum(w_new, target_sum)
    return list(answers), w_new.tolist()

# -------------------------------------------------------------------
# 5) 以 ESS 为目标自动选 gamma（用 prob flatten 的 p^gamma）
#    target_frac: 目标 ESS / n，例如 0.8 表示希望接近 80% 的均匀程度
# -------------------------------------------------------------------
def _ess(p: np.ndarray) -> float:
    p = _renorm_prob(p)
    return 1.0 / float(np.sum(p * p))

def weights_flatten_to_target_ess(
    answers: Sequence[Any],
    weights: Sequence[float],
    target_frac: float = 0.8,
    lam: float = 0.0,
    gamma_lo: float = 0.05,
    gamma_hi: float = 2.0,
    iters: int = 40,
    return_as: str = "sum",  # "sum" or "prob"
) -> Tuple[List[Any], List[float]]:
    _check_len(answers, weights)
    w = _to_np(weights)
    n = len(w)
    target_sum = float(w.sum())
    if n == 0:
        return [], []
    if not (0 < target_frac <= 1.0):
        raise ValueError("target_frac must be in (0,1]")
    target_ess = target_frac * n

    p0 = _renorm_prob(np.maximum(w, 0.0))
    # Binary search gamma so ESS(p^gamma) ~= target_ess
    lo, hi = gamma_lo, gamma_hi

    def make_p(gamma: float) -> np.ndarray:
        p = np.power(np.maximum(p0, 1e-12), gamma)
        p = _renorm_prob(p)
        if lam > 0:
            p = (1 - lam) * p + lam * (1.0 / n)
            p = _renorm_prob(p)
        return p

    # Note: gamma<1 flattens, gamma>1 sharpens
    # ESS is generally decreasing with gamma (sharper => smaller ESS).
    # We'll search for gamma that makes ESS close to target.
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        p_mid = make_p(mid)
        ess_mid = _ess(p_mid)
        if ess_mid > target_ess:
            # too uniform -> increase gamma to sharpen a bit
            lo = mid
        else:
            # too sharp -> decrease gamma
            hi = mid

    gamma_star = (lo + hi) / 2.0
    p_star = make_p(gamma_star)

    if return_as == "prob":
        return list(answers), p_star.tolist()
    elif return_as == "sum":
        return list(answers), (p_star * target_sum).tolist()
    else:
        raise ValueError("return_as must be 'sum' or 'prob'")