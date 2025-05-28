import argparse
import os

import torch
from transformers import PretrainedConfig
from vllm.transformers_utils.config import get_config
from vllm.model_executor.layers.layernorm import RMSNorm

from common import parse_args, ModuleWrapper


class RMSNormWrapper(ModuleWrapper):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.head_size = args.head_size
        self.num_heads = args.num_heads
        self.num_kv_heads = args.num_kv_heads
        self.max_position_embeddings = args.max_position_embeddings
        config = self.get_pretrained_config()
        self.module = RMSNorm(self.head_size, eps=config.rms_norm_eps)
        self.post_init()

    @property
    def input_names(self) -> list[str]:
        return ["x"]

    @property
    def op_prefix(self) -> str:
        return os.path.splitext(os.path.basename(__file__))[0]
    
    def run(self) -> torch.Tensor:
        kwargs = self.make_inputs()
        output = self.module.forward_native(**kwargs)
        return output
    
    def make_inputs(self) -> dict:
        kwargs = {}
        if self.load_inputs:
            kwargs.update(self.load_inputs_from_artifacts())
        else:
            kwargs.update({
                "x": torch.randn(
                    self.seq_len,
                    self.num_heads,
                    self.head_size,
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
    outputs = RMSNormWrapper(args).run()
