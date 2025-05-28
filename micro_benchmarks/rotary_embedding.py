import argparse
import os

import torch
from transformers import PretrainedConfig
from vllm.transformers_utils.config import get_config
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding

from common import parse_args, OpWrapper


class RotaryEmbeddingWrapper(OpWrapper):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        config = self.get_pretrained_config()
        self.head_size = args.head_size
        self.num_heads = args.num_heads
        self.num_kv_heads = args.num_kv_heads
        rope_theta = getattr(config, "rope_theta", 10000)
        self.rotary_embedding = RotaryEmbedding(
            head_size=self.head_size,
            rotary_dim=args.head_size,
            max_position_embeddings=args.max_position_embeddings,
            base=rope_theta,
            is_neox_style=args.is_neox_style,
            dtype=self.torch_dtype,
        )
    
    @property
    def input_names(self) -> list[str]:
        return ["positions", "query", "key"]
    
    @property
    def op_prefix(self) -> str:
        return os.path.splitext(os.path.basename(__file__))[0]
    
    def run(self) -> tuple[torch.Tensor, torch.Tensor]:
        kwargs = self.make_inputs()
        query, key = self.rotary_embedding(**kwargs)
        return query, key
    
    def make_inputs(self) -> dict:
        kwargs = {}
        if self.load_inputs:
            kwargs.update(self.load_inputs_from_artifacts())
        else:
            kwargs.update({
                "positions": torch.arange(self.seq_len, device=self.device, dtype=torch.int64),
                "query": torch.randn(
                    self.seq_len,
                    self.num_heads * self.head_size,
                    device=self.device,
                    dtype=self.torch_dtype
                ),
                "key": torch.randn(
                    self.seq_len,
                    self.num_kv_heads * self.head_size,
                    device=self.device,
                    dtype=self.torch_dtype
                ),
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
    outputs = RotaryEmbeddingWrapper(args).run()
