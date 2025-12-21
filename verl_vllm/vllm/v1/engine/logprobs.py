# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from typing import List, Optional

from vllm.logger import init_logger
from vllm.logprobs import Logprob, PromptLogprobs, SampleLogprobs
from vllm.transformers_utils.detokenizer_utils import (
    AnyTokenizer, convert_ids_list_to_tokens)
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest
from vllm.v1.outputs import LogprobsLists, LogprobsTensors

from scipy.stats import beta
import numpy as np
import random

EPS = 1e-12

@dataclass
class SamplerParams:
    # Beta params for pos/neg
    a_pos: float
    b_pos: float
    a_neg: float
    b_neg: float

    # Prior of positive
    pi: float

    # Acceptance rule a(q) = clip(s0 + scale*s1*q, 0, 1)
    s0: float
    s1: float
    scale: float

    # Targets
    keep_rate: float
    target_pos_in_kept: float

    # For numerical stability
    clip_eps: float = 1e-6


@dataclass
class HistGentleParams:
    bins: int
    q_bins: np.ndarray      # shape [bins], q_bins[b]=P(pos|bin=b)
    w_bins: np.ndarray      # mixture weights for bins, used for calibration

    keep_rate: float        # r
    target_pos_in_kept: float  # t
    lam: float              # λ
    p_min: float
    p_max: float
    delta: float            # δ (calibrated)
    pi: float               # prior used to compute q_bins
    
logger = init_logger(__name__)

NONES = itertools.repeat(None)


@dataclass
class LogprobsProcessor:

    # Tokenizer for this request,
    # None if detokenization is disabled.
    tokenizer: Optional[AnyTokenizer]

    # Logprobs for this request
    logprobs: Optional[SampleLogprobs]
    prompt_logprobs: Optional[PromptLogprobs]
    cumulative_logprob: Optional[float]
    num_logprobs: Optional[int]
    num_prompt_logprobs: Optional[int]
    conf_grouped: float
    
    conf_list: Optional[List[float]]
    conf_group_list: Optional[deque]
    conf_group_size: int
    conf_threshold: Optional[float]
    
    accumulated_token_num: int = 0
    
    probe_max: Optional[float] = -1.0
    probe_min: Optional[float] = -1.0
    
    probe_stop_token_num: int = -1
    probe_stop_has_tested: bool = False
    probe_m: float = 0.5  # default value for probe distribution parameter
    probe_sampler_params: Optional[dict] = None

    @classmethod
    def from_new_request(
        cls,
        tokenizer: Optional[AnyTokenizer],
        request: EngineCoreRequest,
    ) -> "LogprobsProcessor":
        assert request.sampling_params is not None
        num_logprobs = request.sampling_params.logprobs
        num_prompt_logprobs = request.sampling_params.prompt_logprobs
        
        # print('sampling_params', request.sampling_params)
        if hasattr(request.sampling_params, "extra_args") \
        and request.sampling_params.extra_args is not None \
        and request.sampling_params.extra_args.get("enable_conf", False):
            conf_group_size = request.sampling_params.extra_args.get("window_size", 2048)
            conf_threshold  = request.sampling_params.extra_args.get("threshold", 17)
            conf_grouped    = 0.0
            conf_group_list = deque(maxlen=conf_group_size)
            conf_list       = []
        else:
            conf_group_size = -1
            conf_threshold  = None
            conf_grouped    = 0.0
            conf_group_list = None
            conf_list       = None
        
        # assert 0, (request.sampling_params, request.sampling_params.extra_args, request.sampling_params.extra_args.get("probe_max", -1.0), request.sampling_params.extra_args.get("probe_min", -1.0), request.sampling_params.extra_args.get("probe_stop_token_num", -1))
        if hasattr(request.sampling_params, "extra_args") \
            and request.sampling_params.extra_args is not None \
            and request.sampling_params.extra_args.get("probe_max", -1.0) >= 0 \
            and request.sampling_params.extra_args.get("probe_stop_token_num", -1) > 0:
                probe_max = request.sampling_params.extra_args.get("probe_max", -1.0)
                probe_min = request.sampling_params.extra_args.get("probe_min", -1.0)
                probe_stop_token_num = request.sampling_params.extra_args.get("probe_stop_token_num", -1)
                probe_m = request.sampling_params.extra_args.get("probe_m", 0.5)
                # logger.info(f"Probe logits range: max {probe_max}, min {probe_min}")
                # assert 0, (probe_max, probe_min, probe_stop_token_num)
        else:
            probe_max = -1.0
            probe_min = -1.0
            probe_stop_token_num = -1
            probe_m = -1.0
            
        if hasattr(request.sampling_params, "extra_args") \
            and request.sampling_params.extra_args is not None \
            and request.sampling_params.extra_args.get("probe_sampler_params", None) is not None:
                probe_sampler_params = request.sampling_params.extra_args.get("probe_sampler_params", None)
        else:
            probe_sampler_params = None
        # import ipdb; ipdb.set_trace()
    
        return cls(
            tokenizer=tokenizer,
            cumulative_logprob=(None if num_logprobs is None else 0.),
            logprobs=(None if num_logprobs is None else []),
            # NOTE: logprob of first prompt token is None.
            prompt_logprobs=(None if num_prompt_logprobs is None else [None]),
            num_prompt_logprobs=num_prompt_logprobs,
            num_logprobs=num_logprobs,
            
            conf_group_size=conf_group_size,
            conf_grouped=conf_grouped,
            conf_list=conf_list,
            conf_threshold=conf_threshold,
            conf_group_list=conf_group_list,
            
            probe_max=probe_max,
            probe_min=probe_min,
            probe_stop_token_num=probe_stop_token_num,
            probe_m=probe_m,
            probe_sampler_params=probe_sampler_params
        )
        
    ##
    def check_conf_stop(self) -> bool:
        """Return True if the confidence window triggers early stopping."""
        
        # import ipdb; ipdb.set_trace()
        if self.conf_group_list is None or len(self.conf_group_list) == 0:
            return False

        # print('stop')
        # Require a full window; trigger when the moving average is below threshold.
        ret = (len(self.conf_group_list) >= self.conf_group_size
                and self.conf_grouped / len(self.conf_group_list) < self.conf_threshold)
        # if ret:
        #     print(f"Early stopping triggered: conf {self.conf_grouped} / {len(self.conf_group_list):.4f} < threshold {self.conf_threshold} (window size {len(self.conf_group_list)} / {self.conf_group_size})")
        return ret
    
    # def check_probe_stop(self, probe_logits: List[float], new_token_num: int) -> bool:
    #     """Return True if the probe logits trigger early stopping."""
    #     def distribution(self, r, m=0.5):
            
    #         a = self.probe_min   # e.g. 0.12
    #         b = self.probe_max   # e.g. 0.36
    #         # -1是不进行early_exit
    #         if a < 0 or b < 0:
    #             return False
            
    #         import random
    #         assert 0.0 <= r <= 1.0
    #         m = 1-m


            
    #         # 如果区间非法，退化为 p=0.5
    #         if a >= b:
    #             return (random.random() < m)

    #         B = b - a

    #         # 理论上要求 B <= 0.75，否则下面的 p_out 可能为负
    #         # 对你 25%-75% 这种用法，通常 B 会 < 0.75
    #         if B > 0.75:
    #             B = 0.75  # 或者直接 fallback

    #         # K = ∫ bump(r) dr
    #         K = (2.0 / 3.0) * B

    #         # 期望约束: 0.5 = p_out + (1 - p_out)*K
    #         # => p_out = (0.5 - K) / (1 - K)
    #         p_out = (m - K) / (1.0 - K)

    #         # numerical safety
    #         if p_out < 0.0:
    #             p_out = 0.0
    #         elif p_out > 1.0:
    #             p_out = 1.0

    #         # 计算 p(r)
    #         if r < a or r > b:
    #             p = p_out
    #         else:
    #             u = (r - a) / B    # in [0,1]
    #             bump = 4.0 * u * (1.0 - u)   # in [0,1]
    #             p = p_out + (1.0 - p_out) * bump

    #         # p 肯定在 [p_out, 1] ⊆ [0,1]，不需要 clip 了

    #         # 注意：如果你想 "区间内 True 概率大"
    #         # ret=True 的概率就应该是 p，而不是 1-p
    #         ret = (random.random() < p)

    #         return ret

            
    #     if not hasattr(self, 'probe_max') or not hasattr(self, 'probe_min'):
    #         return False
        
    #     self.accumulated_token_num += new_token_num
        
    #     assert len(probe_logits) == 1
        
    #     if not self.probe_stop_has_tested and self.accumulated_token_num >= self.probe_stop_token_num:
    #         self.probe_stop_has_tested = True
    #         ret = distribution(self, probe_logits[0], self.probe_m)
    #         # assert 0, (self.accumulated_token_num, probe_logits[0], self.probe_min, self.probe_max, ret)
    #     else:
    #         ret = False
    #     # assert 0, (probe_logits[0], self.probe_min, self.probe_max, ret)
        
    #     return ret # ret is True means stop generation
    
    def check_probe_stop(self, probe_logits: List[float], new_token_num: int) -> bool:
        """Return True if the probe logits trigger early stopping."""
        
        # def _posterior_q(x: np.ndarray, pi: float, a_pos: float, b_pos: float, a_neg: float, b_neg: float, clip_eps: float) -> np.ndarray:
        #     x = np.asarray(x, dtype=float)
        #     x = np.clip(x, clip_eps, 1 - clip_eps)
        #     f1 = beta.pdf(x, a_pos, b_pos) + EPS
        #     f0 = beta.pdf(x, a_neg, b_neg) + EPS
        #     return (pi * f1) / (pi * f1 + (1 - pi) * f0)
        
        # def accept_probability(
        #     logit: float | np.ndarray,
        #     params: SamplerParams | dict,
        # ) -> float | np.ndarray:
        #     """
        #     Interface #2:
        #     Input: new sample logit + fitted params
        #     Output: pass probability in [0,1]
        #     """
            
        #     if isinstance(params, dict):
        #         params = SamplerParams(**params)

        #     x = np.asarray(logit, dtype=float)
        #     q = _posterior_q([x], params.pi, params.a_pos, params.b_pos, params.a_neg, params.b_neg, clip_eps=params.clip_eps)
        #     p = np.clip(params.s0 + params.scale * params.s1 * q, 0.25, 0.75)
        #     # return scalar if scalar input
        #     return float(p) if np.ndim(logit) == 0 else p
        def _bin_index(x: np.ndarray, bins: int) -> np.ndarray:
            x = np.asarray(x, dtype=float)
            x = np.clip(x, 0.0, 1.0)
            # map [0,1] -> {0,...,bins-1}, 1.0 goes to bins-1
            idx = (x * bins).astype(int)
            return np.clip(idx, 0, bins - 1)
        
        def accept_prob_hist_gentle(logit: float | np.ndarray, params: HistGentleParams | dict) -> float | np.ndarray:
            x = np.asarray(logit, dtype=float)
            if isinstance(params, dict):
                params = HistGentleParams(**params)
            idx = _bin_index(x, params.bins)
            q = params.q_bins[idx]
            p = np.clip(
                params.keep_rate + params.delta + params.lam * (params.target_pos_in_kept - q),
                params.p_min, params.p_max
            )
            # assert 0, (params.lam, params.p_min, params.p_max)
            return float(p) if np.ndim(logit) == 0 else p

            
        if getattr(self, 'probe_sampler_params', None) is None:
            return False
        
        self.accumulated_token_num += new_token_num
        
        assert len(probe_logits) == 1
        
        if not self.probe_stop_has_tested and self.accumulated_token_num >= self.probe_stop_token_num:
            self.probe_stop_has_tested = True
            ret = not (random.random() < accept_prob_hist_gentle(probe_logits[0], self.probe_sampler_params))
            # assert 0, (self.accumulated_token_num, probe_logits[0], self.probe_min, self.probe_max, ret)
        else:
            ret = False
        # assert 0, (probe_logits[0], self.probe_min, self.probe_max, ret)
        
        return ret 

    def _update_sample_logprobs(self, logprobs_lists: LogprobsLists) -> None:
        """Update with sample logprobs from EngineCore.

        Outer lists are only of len > 1 if EngineCore made
        >1 tokens in prior step (e.g. in spec decoding).

        Args:
        logprobs_lists: the lists of logprob tokens, logprobs, and ranks.

        """

        assert self.num_logprobs is not None
        assert self.logprobs is not None
        assert self.cumulative_logprob is not None

        token_ids_lst, logprobs_lst, ranks_lst = logprobs_lists

        for rank, logprobs, token_ids in zip(ranks_lst, logprobs_lst,
                                            token_ids_lst):

            # Detokenize (non-incrementally).
            decoded_tokens = NONES if self.tokenizer is None else (
                convert_ids_list_to_tokens(self.tokenizer, token_ids))

            # Sampler puts the sampled logprob in first.
            sampled_token_logprob = logprobs[0]
            self.cumulative_logprob += sampled_token_logprob

            # Update with the Logprob dictionary for this pos.
            self.logprobs.append(
                self._make_logprob_dict(
                    logprobs,
                    token_ids,
                    decoded_tokens,
                    rank,
                    self.num_logprobs,
                ))
        
        ###
        if self.conf_list is not None:
            # logprobs[0] is the sampled token; use the remaining candidates
            if len(logprobs) > 1:
                new_conf = -sum(logprobs[1:]) / len(logprobs[1:])
            else:
                new_conf = 0.0
            self.conf_list.append(new_conf)

            if len(self.conf_group_list) < self.conf_group_size:
                self.conf_group_list.append(new_conf)
                self.conf_grouped += new_conf
            else:
                self.conf_grouped -= self.conf_group_list.popleft()
                self.conf_group_list.append(new_conf)
                self.conf_grouped += new_conf

    def _update_prompt_logprobs(
        self,
        prompt_logprobs_tensors: LogprobsTensors,
    ) -> None:
        """Update with prompt logprobs from EngineCore.

        Args:
          prompt_logprobs_tensors: tuple containing the prompt logprobs
                                   tensors.

        """

        # Prompt logprobs are enabled.
        assert self.num_prompt_logprobs is not None
        assert self.prompt_logprobs is not None

        token_ids, logprobs, ranks = prompt_logprobs_tensors

        # Detokenize non-incrementally.
        # Output is flat: [num_tok, num_lps] -> [num_tok * num_lps]
        decoded_tokens = None if self.tokenizer is None else (
            convert_ids_list_to_tokens(self.tokenizer,
                                       token_ids.flatten().tolist()))

        # Recover shapes.
        num_prompt_tokens, num_logprobs = logprobs.shape

        # Pythonize the torch tensors.
        prompt_token_ranks = ranks.tolist()
        prompt_logprobs = logprobs.tolist()
        token_ids = token_ids.tolist()

        # Make Logprob for each position.
        for pos in range(num_prompt_tokens):
            # Handle flattening.
            offset = pos * num_logprobs
            offset_end = offset + num_logprobs
            decoded_tokens_for_pos = NONES \
            if decoded_tokens is None else decoded_tokens[offset:offset_end]

            # Update with the Logprob dictionary for this pos.
            self.prompt_logprobs.append(
                self._make_logprob_dict(prompt_logprobs[pos], token_ids[pos],
                                        decoded_tokens_for_pos,
                                        prompt_token_ranks[pos],
                                        self.num_prompt_logprobs))

    def pop_prompt_logprobs(self) -> Optional[PromptLogprobs]:
        """Pop and return all request prompt logprobs

        The logprobs processor aggregates prompt chunk logprobs
        over one or more prefill chunks. This method returns
        all prompt logprobs at once and then forgets them.
        Ensures correct RequestOutputKind.DELTA semantics
        wherein all prompt logprobs are returned at once at
        the end of prefill.

        Returns:
          None if prompt logprobs are disabled for this request.
          List of all prompt logprobs, otherwise.
        """
        plp = self.prompt_logprobs
        if plp:
            self.prompt_logprobs = []
        return plp

    @staticmethod
    def _make_logprob_dict(
        logprobs: list[float],
        logprob_token_ids: list[int],
        decoded_tokens: Iterable[Optional[str]],
        rank: int,
        num_logprobs: int,
    ) -> dict[int, Logprob]:
        """Make a Logprob dictionary for a position.

        Args:
          logprobs: list of log probabilities
          logprob_token_ids: list of top token ids
          decoded_tokens: list of decoded top tokens
          rank: rank of the sampled token
          num_logprobs: number of logprobs requested
            by the user (in addition to sampled logprob)

        Returns:
          dict[token id, Logprob]
        """
        if num_logprobs == -1:
            num_logprobs = len(logprobs)
        # We do not need a special case for the sampled token
        # being in the topk, since inserting duplicated data
        # into a dictionary twice is the same as doing it once.
        topk_ranks = range(1, num_logprobs + 1)
        ranks = itertools.chain((rank, ), topk_ranks)

        return {
            token_id: Logprob(
                logprob=logprob,
                rank=rank,
                decoded_token=token,
            )
            for token_id, logprob, rank, token in zip(
                logprob_token_ids, logprobs, ranks, decoded_tokens)
        }

    def update_from_output(self, output: EngineCoreOutput) -> None:
        if output.new_logprobs is not None:
            self._update_sample_logprobs(output.new_logprobs)
        if output.new_prompt_logprobs_tensors is not None:
            self._update_prompt_logprobs(output.new_prompt_logprobs_tensors)
