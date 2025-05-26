import argparse
import os

import torch
from vllm import _custom_ops as ops

from common import parse_args, OpWrapper


class TopkSoftmaxWrapper(OpWrapper):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.renormalize_topk_softmax = args.renormalize_topk_softmax
        self.enforce_fp32_router = self.device == "cuda" # vllm topk_softmax is implemented in fp32.
        self.seq_len = args.seq_len
        self.num_experts = args.num_experts
        self.topk = args.topk
    
    @property
    def input_names(self) -> list[str]:
        return ["gating_output"]
    
    @property
    def op_prefix(self) -> str:
        return os.path.splitext(os.path.basename(__file__))[0]

    def run(self) -> tuple[torch.Tensor, torch.Tensor]:
        kwargs = self.make_inputs()
        ops.topk_softmax(**kwargs) # inplace operation
        topk_weights = kwargs["topk_weights"]
        topk_indices = kwargs["topk_ids"]
        if self.renormalize_topk_softmax:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        return topk_weights, topk_indices
    
    def make_inputs(self) -> dict:
        kwargs = {}
        if self.load_inputs:
            kwargs.update(self.load_inputs_from_artifacts())
        else:
            kwargs.update({
                "gating_output": torch.randn(
                    self.seq_len,
                    self.num_experts,
                    device=self.device,
                    dtype=torch.float32 if self.enforce_fp32_router else self.torch_dtype,
                ),
            })
        
        topk_weights = torch.empty(
            self.seq_len,
            self.topk,
            device=self.device,
            dtype=torch.float32 if self.enforce_fp32_router else self.torch_dtype
        )
        topk_indices = torch.empty(
            self.seq_len,
            self.topk,
            device=self.device,
            dtype=torch.int32
        )
        token_expert_indices = torch.empty(
            self.seq_len,
            self.topk,
            device=self.device,
            dtype=torch.int32
        )
        kwargs.update({
            "topk_weights": topk_weights,
            "topk_ids": topk_indices,
            "token_expert_indicies": token_expert_indices,
        })

        if self.save_inputs:
            self.save_inputs_to_artifacts(kwargs)
        return kwargs
        

if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    outputs = TopkSoftmaxWrapper(args).run()
