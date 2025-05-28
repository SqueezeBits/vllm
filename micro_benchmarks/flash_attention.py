import argparse
import os
from typing import Callable

import torch
from vllm.attention.backends.flash_attn import FlashAttentionImpl
from vllm.attention.layer import Attention

from common import parse_args, OpWrapper


class FlashAttentionWrapper(OpWrapper):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.seq_len = args.seq_len
        self.hidden_dim = args.hidden_dim
        self.head_size = args.head_size
        self.num_heads = args.num_heads
        if self.device == "cuda":
            self.attn_impl = FlashAttentionImpl(
                num_heads=self.num_heads,
                head_size=self.head_size,
                scale=self.head_size**-0.5,
                num_kv_heads=self.num_heads,
                alibi_slopes=None,
                sliding_window=None,
                kv_cache_dtype="auto",
            )
        else:
            raise NotImplementedError(f"FlashAttentionImpl for {self.device} is not implemented yet.")
        self.attention_layer = Attention(self.num_heads, self.head_size, 1.0)
    
    @property
    def input_names(self) -> list[str]:
        return ["query", "key", "value", "kv_cache"]
    
    @property
    def op_prefix(self) -> str:
        return os.path.splitext(os.path.basename(__file__))[0]
    
    @property
    def target_func(self) -> Callable:
        if self.device == "cuda":
            return self.attn_impl.forward
        else:
            raise NotImplementedError(f"FlashAttentionImpl for {self.device} is not implemented yet.")
    
    def run(self) -> torch.Tensor:
        kwargs = self.make_inputs()
        attn_metadata = self.build_attn_metadata()
        output = torch.empty(
            self.seq_len,
            self.num_heads,
            self.head_size,
            device=self.device,
            dtype=self.torch_dtype
        )
        kwargs.update({
            "attn_metadata": attn_metadata,
            "output": output
        })
        self.target_func(self.attention_layer, **kwargs)
        return output
    
    def make_inputs(self) -> dict:
        kwargs = {}
        if self.load_inputs:
            kwargs.update(self.load_inputs_from_artifacts())
        else:
            kwargs.update({
                "query": torch.randn(
                    self.seq_len,
                    self.num_heads,
                    self.head_size,
                    device=self.device,
                    dtype=self.torch_dtype
                ),
                "key": torch.randn(
                    self.seq_len,
                    self.num_heads,
                    self.head_size,
                    device=self.device,
                    dtype=self.torch_dtype
                ),
                "value": torch.randn(
                    self.seq_len,
                    self.num_heads,
                    self.head_size,
                    device=self.device,
                    dtype=self.torch_dtype
                ),
                "kv_cache": torch.empty(0, device=self.device, dtype=self.torch_dtype)
            })
        
        if self.save_inputs:
            self.save_inputs_to_artifacts(kwargs)
        return kwargs
    

if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    output = FlashAttentionWrapper(args).run()
