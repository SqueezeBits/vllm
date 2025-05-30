import argparse
import os

import torch
from transformers import PretrainedConfig
from vllm.transformers_utils.config import get_config
from vllm.model_executor.model_loader.loader import set_current_vllm_config
from vllm.model_executor.models.qwen3_moe import Qwen3MoeDecoderLayer
from vllm.forward_context import set_forward_context
from vllm.config import VllmConfig

from common import parse_args, ModuleWrapper


class Qwen3DecoderLayerWrapper(ModuleWrapper):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.prefix = "model.layers.0"
        with set_current_vllm_config(super().build_vllm_config()):
            self.module = Qwen3MoeDecoderLayer(
                config=self.get_pretrained_config(),
                prefix=self.prefix
            )
            if self.ep_size > 1:
                self.device = f"cuda:{args.rank}"
                torch.cuda.set_device(torch.device(self.device))
            self.post_init()

    @property
    def input_names(self) -> list[str]:
        return ["positions", "hidden_states"]
    
    @property
    def op_prefix(self) -> str:
        return os.path.splitext(os.path.basename(__file__))[0]

    def run(self) -> tuple[torch.Tensor, torch.Tensor]:
        kwargs = self.make_inputs()
        attn_metadata = self.build_attn_metadata()
        vllm_config = self.build_vllm_config()
        with set_forward_context(attn_metadata, vllm_config):
            hidden_states, residual = self.module.forward(**kwargs)
        return hidden_states, residual

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
                "residual": None
            })
        
        if self.save_inputs:
            self.save_inputs_to_artifacts(kwargs)
        return kwargs
    
    def build_vllm_config(self) -> VllmConfig:
        vllm_config = super().build_vllm_config()
        vllm_config.compilation_config.static_forward_context = {
            f"{self.prefix}.self_attn.attn": self.module.self_attn.attn,
        }
        return vllm_config
    
    def post_init(self):
        super().post_init()
        if self.ep_size > 1:
            self.module.mlp.experts.expert_map = self.module.mlp.experts.expert_map.to(self.device)
    
    def get_pretrained_config(self) -> PretrainedConfig:
        hf_config = get_config("Qwen/Qwen3-30B-A3B", False)
        return hf_config


def run_rank(rank, args):
    """Run the test for a specific rank."""
    args.rank = rank
    outputs = Qwen3DecoderLayerWrapper(args).run()

if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.ep_size == 1:
        output = Qwen3DecoderLayerWrapper(args).run()
    else:
        torch.multiprocessing.spawn(
            run_rank,
            args=(args,),
            nprocs=args.ep_size,
            join=True
        )
