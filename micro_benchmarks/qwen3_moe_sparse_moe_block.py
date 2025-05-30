import argparse
import os
from typing import Iterable

import torch
from transformers import PretrainedConfig
from vllm.transformers_utils.config import get_config
from vllm.model_executor.model_loader.utils import set_default_torch_dtype
from vllm.model_executor.model_loader.loader import set_current_vllm_config
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.models.qwen3_moe import Qwen3MoeSparseMoeBlock
from vllm.forward_context import set_forward_context

from common import parse_args, ModuleWrapper


class Qwen3MoeSparseMoeBlockWrapper(ModuleWrapper):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.prefix = "model.layers.0.mlp"
        self.num_experts = args.num_experts
        if self.ep_size > 1:
            self.device = f"cuda:{args.rank}"
            torch.cuda.set_device(torch.device(self.device))
        with set_current_vllm_config(super().build_vllm_config()), \
                torch.device(self.device), \
                set_default_torch_dtype(self.torch_dtype):
            self.module = Qwen3MoeSparseMoeBlock(
                config=self.get_pretrained_config(),
                prefix=self.prefix
            )
            self.initialize_weights()
    
    @property
    def input_names(self) -> list[str]:
        return ["hidden_states"]
    
    @property
    def op_prefix(self) -> str:
        return os.path.splitext(os.path.basename(__file__))[0]
    
    @property
    def model(self) -> str:
        return "Qwen/Qwen3-30B-A3B"
    
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
        hf_config = get_config(self.model, False)
        return hf_config
    
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.num_experts)
        params_dict = {f"{self.prefix}.{name}": param for name, param in self.module.named_parameters()}
        for name, loaded_weight in weights:
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if ((name.endswith(".bias") or name.endswith("_bias"))
                        and name not in params_dict):
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    # Skip loading extra bias for GPTQ models.
                    if ((name.endswith(".bias") or name.endswith("_bias"))
                            and name not in params_dict):
                        continue
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param,
                                  loaded_weight,
                                  name,
                                  shard_id=shard_id,
                                  expert_id=expert_id)
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if ((name.endswith(".bias") or name.endswith("_bias"))
                            and name not in params_dict):
                        continue
                    # Remapping the name of FP8 kv-scale.
                    if name.endswith("kv_scale"):
                        remapped_kv_scale_name = name.replace(
                            ".kv_scale", ".attn.kv_scale")
                        if remapped_kv_scale_name not in params_dict:
                            continue
                        else:
                            name = remapped_kv_scale_name
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
        return

def run_rank(rank, args):
    """Run the test for a specific rank."""
    args.rank = rank
    outputs = Qwen3MoeSparseMoeBlockWrapper(args).run()


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    if args.ep_size == 1:
        output = Qwen3MoeSparseMoeBlockWrapper(args).run()
    else:
        torch.multiprocessing.spawn(
            run_rank,
            args=(args,),
            nprocs=args.ep_size,
        )
