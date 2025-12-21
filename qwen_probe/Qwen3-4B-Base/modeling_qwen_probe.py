# modeling_qwen3_probe.py

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

import logging
import sys
def setup_logger(name: str = __name__) -> logging.Logger:
    fmt = "%(asctime)s [%(levelname)s] [%(process)d:%(threadName)s] %(name)s: %(message)s"
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt))
    logger = logging.getLogger(name)
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)
    logger.propagate = False  # avoid double logs if root has handlers
    return logger

logger = setup_logger()


@dataclass
class CausalLMOutputWithPastAndProbe(CausalLMOutputWithPast):
    probe_logits: Optional[torch.FloatTensor] = None
    probe_loss: Optional[torch.FloatTensor] = None

class Qwen3ProbeForCausalLM(Qwen3ForCausalLM):
    # Set auto class so Transformers/Hub saving does not inject a None key into auto_map.
    _auto_class = "AutoModelForCausalLM"
    def __init__(self, config):
        super().__init__(config)
        # self.probe_head = nn.Linear(config.hidden_size, 1)
        
        self.probe_head = nn.Sequential(
            nn.Linear(config.hidden_size, 128),
            nn.ReLU(),              # 也可以换成 nn.ReLU()
            nn.Linear(128, 1),
        )
        # Initialize probe head explicitly when loading from base Qwen3 (weights won't exist).
        # meta tensors can't host a generator; fall back to CPU in that case
        if isinstance(self.probe_head, nn.Linear):
            _gen_device = "cpu" if self.probe_head.weight.device.type == "meta" else self.probe_head.weight.device
        else:
            _gen_device = "cpu" if next(self.probe_head.parameters()).device.type == "meta" else next(self.probe_head.parameters()).device
        _probe_gen = torch.Generator(device=_gen_device)
        _probe_gen.manual_seed(getattr(config, "probe_init_seed", 1234))
        
        for m in self.probe_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)   # 适合 GELU/ReLU 的通用选择
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # if hasattr(config, "initializer_range"):
        #     # nn.init.normal_(self.probe_head.weight, mean=0.0, std=config.initializer_range, generator=_probe_gen)
        #     nn.init.zeros_(self.probe_head.weight)
        # else:
        #     # nn.init.xavier_uniform_(self.probe_head.weight, generator=_probe_gen)
        #     nn.init.zeros_(self.probe_head.weight)
        # if self.probe_head.bias is not None:
        #     nn.init.zeros_(self.probe_head.bias)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        probe_labels: Optional[torch.Tensor] = None,
        early_exit: Optional[torch.Tensor] = None,
        probe_stop_token_num: int = -1,
        # compute_probe: bool = False,
        **kwargs,
    ):
        return_dict = kwargs.pop("return_dict", self.config.use_return_dict)

        # 一定要拿 hidden_states
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,  # force dict to make it easy to attach probe output
            **kwargs,
        )
        
        probe_logits_ = None
        probe_loss = None
        # if compute_probe or probe_labels is not None:
            # last_hidden: [bs, seq_len, hidden_size]
        if kwargs.get('pad_zero_num', None) is not None:
            # assert 0, kwargs['pad_zero_num'].tolist()
            last_hidden = outputs.hidden_states[-1]
            
            if early_exit is not None:
                early_exit = ~(early_exit.bool())

            # 例如只对最后一个非 padding token 做 probe
            if attention_mask is not None:
                # seq_len = attention_mask.sum(dim=-1) - 1  # last valid position
                seq_len = attention_mask.shape[1] - kwargs['pad_zero_num'] - 1
                seq_len = seq_len.clamp(min=0)
                probe_input = last_hidden.gather(
                    1, seq_len.view(-1, 1, 1).expand(-1, 1, last_hidden.size(-1))
                ).squeeze(1)  # [bs, hidden]
                # assert 0, (seq_len, probe_input)
            else:
                # probe_input = last_hidden[:, -1, :]       # [bs, hidden]
                raise NotImplementedError("attention_mask is required for probe when pad tokens exist.")
            if early_exit is not None:
                probe_input = probe_input[early_exit]
            if probe_stop_token_num > 0:
                
                start_pos = attention_mask.sum(dim=-1) - 2 + kwargs['pad_zero_num']
                effective_len = attention_mask.sum(dim=-1) - 2
                effective_len = torch.clamp(effective_len, max=probe_stop_token_num, min=0)
                stop_pos = attention_mask.shape[1] - start_pos + effective_len - 1
                
                stop_probe = last_hidden.gather(
                    1, stop_pos.view(-1, 1, 1).expand(-1, 1, last_hidden.size(-1))
                ).squeeze(1)
                if early_exit is not None:
                    stop_probe = stop_probe[early_exit]
                probe_input = torch.cat([stop_probe, probe_input], dim=0)
                
                
            probe_input = probe_input.detach()        # 不让 probe_loss 回传到 backbone

            probe_logits = self.probe_head(probe_input).squeeze(-1)  # [bs]
            probe_logits_ = torch.sigmoid(probe_logits)
            # if probe_stop_token_num > 0:
            #     probe_logits_ = probe_logits_[probe_logits_.size(0) // 2: ]
                
            if probe_labels is not None:
                probe_labels = probe_labels.float().view(-1)
                if early_exit is not None:
                    probe_labels = probe_labels[early_exit]
                if probe_labels.shape[0] <= 0:
                    probe_loss = None
                else:
                    if probe_stop_token_num > 0:
                        probe_labels = torch.cat([probe_labels, probe_labels], dim=0)
                    probe_loss = F.binary_cross_entropy_with_logits(
                        probe_logits.view(-1),
                        probe_labels,
                    )
            else:
                probe_loss = None
                
            # if len(probe_logits_.tolist())>=2:
            #     if abs(probe_logits_.tolist()[0]) > 1 - 1e-10:
            #         assert 0, (probe_logits_.tolist(), attention_mask[0].tolist(), input_ids[0].tolist(), kwargs['pad_zero_num'], attention_mask.sum(dim=-1), attention_mask.shape[1], seq_len, stop_pos)
            # assert 0, (probe_logits_.tolist(), kwargs['pad_zero_num'], attention_mask.sum(dim=-1), attention_mask.shape[1], seq_len, stop_pos)
        if not return_dict:
            # keep tuple order consistent with `CausalLMOutputWithPast`
            return outputs.to_tuple() + (probe_logits_, probe_loss)

        return CausalLMOutputWithPastAndProbe(
            loss=outputs.loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            probe_logits=probe_logits_,
            probe_loss=probe_loss,
        )
