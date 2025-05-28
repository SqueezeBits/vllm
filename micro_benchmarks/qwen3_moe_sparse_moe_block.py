import argparse
import os

import torch
from transformers import PretrainedConfig
from vllm.transformers_utils.config import get_config
from vllm.model_executor.models.qwen3_moe import Qwen3MoeSparseMoeBlock
from vllm.forward_context import set_forward_context

from common import parse_args, ModuleWrapper


class Qwen3MoeSparseMoeBlockWrapper(ModuleWrapper):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.prefix = "model.layers.0"
        self.module = Qwen3MoeSparseMoeBlock(
            config=self.get_pretrained_config(),
            prefix=self.prefix
        )
        self.post_init()
    
    @property
    def input_names(self) -> list[str]:
        return ["hidden_states"]
    
    @property
    def op_prefix(self) -> str:
        return os.path.splitext(os.path.basename(__file__))[0]
    
    def run(self) -> torch.Tensor:
        kwargs = self.make_inputs()
        attn_metadata = self.build_attn_metadata()
        vllm_config = self.build_vllm_config()
        with set_forward_context(attn_metadata, vllm_config):
            hidden_states = self.module.forward(**kwargs)
        return hidden_states
    
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
                )
            })
        
        if self.save_inputs:
            self.save_inputs_to_artifacts(kwargs)
        return kwargs
    
    def get_pretrained_config(self) -> PretrainedConfig:
        hf_config = get_config("Qwen/Qwen3-30B-A3B", False)
        return hf_config


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    output = Qwen3MoeSparseMoeBlockWrapper(args).run()
