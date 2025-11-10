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

from vllm.sampling_params import SamplingParams
from enum import Enum, auto
import re

logger = init_logger(__name__)

NONES = itertools.repeat(None)

class StopState(Enum):
    NOT_BRACKETED = auto()
    IN_START = auto()
    IN_BRACKET = auto()
    IN_END = auto()
    OUT_BRACKET = auto()
    step = -1


@dataclass
class StopProcessor:
    # Tokenizer for this request,
    # None if detokenization is disabled.
    tokenizer: Optional[AnyTokenizer]
    samping_params: Optional[SamplingParams]
    boxed_stop: bool
    boxed_stop_state: Optional[StopState]
    boxed_stop_idx: int
    # start_stop_tokens: List[int]
    # end_stop_tokens: List[int]
    start_stop_tokens: str
    end_stop_tokens: str
    
    @classmethod
    def from_new_request(
        cls,
        tokenizer: Optional[AnyTokenizer],
        request: EngineCoreRequest,
    ) -> "StopProcessor":
        # raise NotImplementedError
        assert request.sampling_params is not None
        sampling_params = request.sampling_params
        if hasattr(request.sampling_params, "stop_at_boxed") \
            and request.sampling_params.stop_at_boxed:
                boxed_stop = True
                boxed_stop_state = StopState.NOT_BRACKETED
                boxed_stop_idx = -1
                # start_stop_tokens = tokenizer.encode("$\\boxed{")
                # end_stop_tokens = tokenizer.encode("}$")
                start_stop_tokens = "$\\boxed{"
                end_stop_tokens = "}$"
                
        else:
            boxed_stop = False
            boxed_stop_state = None
            boxed_stop_idx = -1
            start_stop_tokens = []
            end_stop_tokens = []

        return cls(
            tokenizer=tokenizer,
            samping_params=sampling_params,
            boxed_stop=boxed_stop,
            boxed_stop_state=boxed_stop_state,
            boxed_stop_idx=boxed_stop_idx,
            start_stop_tokens=start_stop_tokens,
            end_stop_tokens=end_stop_tokens,
        )
    
    def should_stop(self, output: EngineCoreOutput) -> bool:
        if self.boxed_stop_state is None:
            return False
        current_token_id = output.new_token_ids[0]
        
        # # assert 0, (self.start_stop_tokens, self.end_stop_tokens)
        # if self.boxed_stop_state == StopState.NOT_BRACKETED:
        #     if current_token_id == self.start_stop_tokens[0]:
        #         assert 0, (current_token_id, self.start_stop_tokens, self.end_stop_tokens)
        #         self.boxed_stop_state = StopState.IN_START
        #         self.boxed_stop_idx = 1
        #         if self.boxed_stop_idx == len(self.start_stop_tokens):
        #             self.boxed_stop_state = StopState.IN_BRACKET
        #             self.boxed_stop_idx = -1
        # elif self.boxed_stop_state == StopState.IN_START:
        #     if current_token_id == self.start_stop_tokens[self.boxed_stop_idx]:
        #         self.boxed_stop_idx += 1
        #         if self.boxed_stop_idx == len(self.start_stop_tokens):
        #             self.boxed_stop_state = StopState.IN_BRACKET
        #             self.boxed_stop_idx = -1
        #     else:
        #         self.boxed_stop_state = StopState.NOT_BRACKETED
        #         self.boxed_stop_idx = -1
                    
        # elif self.boxed_stop_state == StopState.IN_BRACKET:
        #     if current_token_id == self.end_stop_tokens[0]:
        #         self.boxed_stop_idx = 1
        #         if self.boxed_stop_idx == len(self.start_stop_tokens):
        #             self.boxed_stop_state = StopState.OUT_BRACKET
        #             self.boxed_stop_idx = 0
        #             return True
        # elif self.boxed_stop_state == StopState.IN_END:
        #     if current_token_id == self.end_stop_tokens[self.boxed_stop_idx]:
        #         self.boxed_stop_idx += 1
        #         if self.boxed_stop_idx == len(self.end_stop_tokens):
        #             self.boxed_stop_state = StopState.OUT_BRACKET
        #             self.boxed_stop_idx = 0
        #             return True
        #     else:
        #         self.boxed_stop_state = StopState.IN_BRACKET
        #         self.boxed_stop_idx = -1
                
                
        # elif self.boxed_stop_state == StopState.OUT_BRACKET:
        #     raise ValueError("Should not reach here.")
        current_string = self.tokenizer.decode([current_token_id])
        current_string = re.sub(r'\s+', '', current_string)
        if self.boxed_stop_state == StopState.NOT_BRACKETED:
            if self.start_stop_tokens.startswith(current_string):
                if current_string >= self.start_stop_tokens:
                    self.boxed_stop_state = StopState.IN_BRACKET
                else:
                    self.boxed_stop_idx = len(current_string)
                    self.boxed_stop_state = StopState.IN_START
        elif self.boxed_stop_state == StopState.IN_START:
            expected_string = self.start_stop_tokens[self.boxed_stop_idx:]
            if expected_string.startswith(current_string):
                self.boxed_stop_idx += len(current_string)
                if self.boxed_stop_idx >= len(self.start_stop_tokens):
                    self.boxed_stop_state = StopState.IN_BRACKET
                    self.boxed_stop_idx = -1
            else:
                self.boxed_stop_state = StopState.NOT_BRACKETED
                self.boxed_stop_idx = -1
        elif self.boxed_stop_state == StopState.IN_BRACKET:
            if self.end_stop_tokens.startswith(current_string):
                if current_string >= self.end_stop_tokens:
                    self.boxed_stop_state = StopState.OUT_BRACKET
                    return True
                else:
                    self.boxed_stop_idx = len(current_string)
                    self.boxed_stop_state = StopState.IN_END
        elif self.boxed_stop_state == StopState.IN_END:
            expected_string = self.end_stop_tokens[self.boxed_stop_idx:]
            if expected_string.startswith(current_string):
                self.boxed_stop_idx += len(current_string)
                if self.boxed_stop_idx >= len(self.end_stop_tokens):
                    self.boxed_stop_state = StopState.OUT_BRACKET
                    return True
            else:
                self.boxed_stop_state = StopState.NOT_BRACKETED
                self.boxed_stop_idx = -1
        
        return False