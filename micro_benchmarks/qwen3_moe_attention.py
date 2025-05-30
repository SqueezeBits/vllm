import argparse
import os

import torch
from transformers import PretrainedConfig
from vllm.transformers_utils.config import get_config
from vllm.model_executor.model_loader.utils import set_default_torch_dtype
from vllm.model_executor.model_loader.loader import set_current_vllm_config
from vllm.model_executor.models.qwen3_moe import Qwen3MoeAttention
from vllm.forward_context import set_forward_context
from vllm.config import VllmConfig

from common import parse_args, ModuleWrapper


class Qwen3MoeAttentionWrapper(ModuleWrapper):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.prefix = "model.layers.0"
        config = self.get_pretrained_config()
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        with set_current_vllm_config(super().build_vllm_config()), \
                torch.device(self.device), \
                set_default_torch_dtype(self.torch_dtype):
            self.module = Qwen3MoeAttention(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                rope_theta=rope_theta,
                rope_scaling=rope_scaling,
                max_position_embeddings=max_position_embeddings,
                rms_norm_eps=config.rms_norm_eps,
                qkv_bias=getattr(config, 'attention_bias', False),
                head_dim=getattr(config, 'head_dim', None),
                prefix=f"{self.prefix}.self_attn",
            )
            self.initialize_weights()
    
    @property
    def input_names(self) -> list[str]:
        return ["positions", "hidden_states"]
    
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
                "positions": torch.arange(self.seq_len, device=self.device, dtype=torch.int64),
                "hidden_states": torch.randn(
                    self.seq_len,
                    self.hidden_dim,
                    device=self.device,
                    dtype=self.torch_dtype
                ),
            })
        
        if self.save_inputs:
            self.save_inputs_to_artifacts(kwargs)
        return kwargs
    
    def build_vllm_config(self) -> VllmConfig:
        vllm_config = super().build_vllm_config()
        vllm_config.compilation_config.static_forward_context = {
            f"{self.prefix}.self_attn.attn": self.module.attn,
        }
        return vllm_config
    
    def get_pretrained_config(self) -> PretrainedConfig:
        hf_config = get_config("Qwen/Qwen3-30B-A3B", False)
        return hf_config


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    output = Qwen3MoeAttentionWrapper(args).run()
