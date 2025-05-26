import argparse
import os

import torch
from vllm.model_executor.layers.fused_moe.fused_moe import torch_vllm_inplace_fused_experts

from common import parse_args, OpWrapper


class FusedExpertsWrapper(OpWrapper):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.seq_len = args.seq_len
        self.hidden_dim = args.hidden_dim
        self.num_experts = args.num_experts
        self.ep_size = args.ep_size
        self.moe_intermediate_size = args.moe_intermediate_size
        self.topk = args.topk
        assert self.num_experts % self.ep_size == 0, \
            f"num_experts ({self.num_experts}) is not divisible by ep size ({self.ep_size})."
    
    @property
    def input_names(self) -> list[str]:
        return [
            "hidden_states",
            "w1",
            "w2",
            "topk_weights",
            "topk_ids",
        ]
    
    @property
    def op_prefix(self) -> str:
        return os.path.splitext(os.path.basename(__file__))[0]

    def run(self) -> torch.Tensor:
        kwargs = self.make_inputs()
        output = torch_vllm_inplace_fused_experts(**kwargs)
        return output
    
    def make_inputs(self) -> dict:
        kwargs = {}
        if self.load_inputs:
            kwargs.update(self.load_inputs_from_artifacts())
        else:
            kwargs.update({
                "hidden_states": torch.randn(
                    self.seq_len,
                    self.hidden_dim,
                    device=self.device,
                    dtype=self.torch_dtype
                ),
                "w1": torch.randn(
                    self.num_experts//self.ep_size,
                    self.moe_intermediate_size * 2,
                    self.hidden_dim,
                    device=self.device,
                    dtype=self.torch_dtype,
                ),
                "w2": torch.randn(
                    self.num_experts//self.ep_size,
                    self.hidden_dim,
                    self.moe_intermediate_size,
                    device=self.device,
                    dtype=self.torch_dtype,
                ),
                "topk_weights": torch.randn(
                    self.seq_len,
                    self.topk,
                    device=self.device,
                    dtype=self.torch_dtype,
                ),
                "topk_ids": torch.randint(
                    0,
                    self.num_experts//self.ep_size,
                    (self.seq_len, self.topk),
                    device=self.device,
                    dtype=torch.int32,
                ),
            })
        
        expert_map = self.create_expert_map()
        kwargs.update({
            "global_num_experts": self.num_experts,
            "expert_map": expert_map,
            "activation": "silu",
            "apply_router_weight_on_input": False,
            "use_fp8_w8a8": False,
            "use_int8_w8a8": False,
            "use_int8_w8a16": False,
            "use_int4_w4a16": False,
            "per_channel_quant": False,
            "w1_scale": None,
            "w2_scale": None,
            "w1_zp": None,
            "w2_zp": None,
            "a1_scale": None,
            "a2_scale": None,
            "block_shape": None,
        })

        if self.save_inputs:
            self.save_inputs_to_artifacts(kwargs)
        return kwargs
    
    def create_expert_map(self, rank: int = 0) -> torch.Tensor | None:
        if self.ep_size == 1:
            return None
        else:
            experts_per_device = self.num_experts // self.ep_size
            expert_map = torch.empty(self.num_experts, device=self.device, dtype=torch.int32).fill_(-1)
            cur_expert_map = torch.arange(experts_per_device, device=self.device, dtype=torch.int32)
            start_idx = rank * experts_per_device
            end_idx = start_idx + experts_per_device
            expert_map[start_idx:end_idx] = cur_expert_map
            return expert_map


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    output = FusedExpertsWrapper(args).run()
