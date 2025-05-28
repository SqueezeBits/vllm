from abc import ABC, abstractmethod
import argparse
import os
from typing import Any

import torch
from vllm.distributed.parallel_state import init_distributed_environment, initialize_model_parallel
from vllm.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.config import VllmConfig, ParallelConfig, CompilationConfig
from vllm.utils import get_distributed_init_method, get_ip, get_open_port


dtype_map = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--distributed-init-method", type=str, default=get_distributed_init_method(get_ip(), get_open_port()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--seq_len", type=int, default=8192)
    parser.add_argument("--hidden_dim", type=int, default=2048)
    parser.add_argument("--num_experts", type=int, default=128)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--moe-intermediate-size", type=int, default=768)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--ep-size", type=int, default=1)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--save-inputs", action="store_true", help="Save input tensors to reuse in future runs.")
    parser.add_argument("--load-inputs", action="store_true", help="Load saved input tensors.")
    parser.add_argument("--artifacts-dir", type=str, default=os.path.join(os.path.dirname(__file__), "artifacts"))
    parser.add_argument("--renormalize-topk-softmax", action="store_true")
    args = parser.parse_args()
    if args.save_inputs:
        os.makedirs(args.artifacts_dir, exist_ok=True)
    return parser.parse_args()


class OpWrapper(ABC):
    def __init__(self, args: argparse.Namespace):
        self.device = args.device
        self.torch_dtype = dtype_map[args.dtype]
        self.save_inputs = args.save_inputs
        self.load_inputs = args.load_inputs
        self.artifacts_dir = args.artifacts_dir
    
    @property
    @abstractmethod
    def input_names(self) -> list[str]:
        pass

    @property
    @abstractmethod
    def op_prefix(self) -> str:
        pass

    @abstractmethod
    def run(self) -> Any:
        pass

    def save_inputs_to_artifacts(self, inputs: dict) -> None:
        for input_name in self.input_names:
            torch.save(inputs[input_name], os.path.join(self.artifacts_dir, f"{self.op_prefix}_input_{input_name}.pt"))
    
    def load_inputs_from_artifacts(self) -> dict:
        inputs_loaded = {}
        for input_name in self.input_names:
            saved_path = os.path.join(self.artifacts_dir, f"{self.op_prefix}_input_{input_name}.pt")
            if not os.path.exists(saved_path):
                raise FileNotFoundError(f"Input file {saved_path} not found.")
            inputs_loaded[input_name] = torch.load(saved_path)
        return inputs_loaded


class ModuleWrapper(OpWrapper):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.seq_len = args.seq_len
        self.hidden_dim = args.hidden_dim
        self.tp_size = args.tp_size
        self.rank = args.rank
        self.initialize_parallel_state(args)
        torch.set_default_dtype(self.torch_dtype)
    
    def initialize_parallel_state(self, args: argparse.Namespace) -> None:
        backend = "nccl" if self.device == "cuda" else None
        if backend is None:
            raise NotImplementedError(f"Backend {backend} is not supported yet.")

        init_distributed_environment(
            world_size=args.ep_size,
            rank=self.rank,
            local_rank=self.rank,
            backend=backend,
            distributed_init_method=args.distributed_init_method,
        )
        # GroupCoordinator is initialized based on current platform.
        initialize_model_parallel(args.tp_size)

    def post_init(self) -> None:
        assert hasattr(self, "module"), "Module is not initialized yet."
        self.initialize_weights()
        self.module.to(self.device).to(self.torch_dtype)
    
    def initialize_weights(self) -> None:
        # TODO: enable loading pretrained weights
        with torch.no_grad():
            for _, param_value in self.module.named_parameters():
                param_value.data.copy_(torch.randn_like(param_value))
    
    def build_attn_metadata(self) -> FlashAttentionMetadata:
        if self.device == "cuda":
            # FlashAttentionMetadata
            seq_block_size = 32
            seq_block_num = self.seq_len // seq_block_size
            assert self.seq_len % seq_block_size == 0
            seq_lens = [seq_block_size] * seq_block_num
            seq_lens_tensor = torch.tensor(seq_lens, device=self.device, dtype=torch.int32)
            max_prefill_seq_len = seq_block_size
            max_decode_seq_len = 0
            context_lens_tensor = torch.tensor([0] * seq_block_num, device=self.device, dtype=torch.int32)
            block_tables = torch.empty(seq_block_num, 0, device=self.device, dtype=torch.int32)
            use_cuda_graph = False
            max_query_len = 32
            max_decode_query_len = 1
            query_start_loc = torch.arange(0, self.seq_len, seq_block_size, device=self.device, dtype=torch.int32)
            seq_start_loc = torch.arange(0, self.seq_len, seq_block_size, device=self.device, dtype=torch.int32)
            encoder_seq_lens = None
            encoder_seq_lens_tensor = None
            encoder_seq_start_loc = None
            max_encoder_seq_len = None
            num_encoder_tokens = None
            cross_slot_mapping = None
            cross_block_tables = None
            # TODO: check _cached_metadata

            # AttentionMetadata
            num_prefills = seq_block_num
            num_prefill_tokens = self.seq_len
            num_decode_tokens = 0
            slot_mapping = torch.tensor([-1] * self.seq_len, device=self.device, dtype=torch.int64)
            multi_modal_placeholder_index_maps = dict()
            enable_kv_scales_calculation = False

            return FlashAttentionMetadata(
                seq_lens=seq_lens,
                seq_lens_tensor=seq_lens_tensor,
                max_prefill_seq_len=max_prefill_seq_len,
                max_decode_seq_len=max_decode_seq_len,
                context_lens_tensor=context_lens_tensor,
                block_tables=block_tables,
                use_cuda_graph=use_cuda_graph,
                max_query_len=max_query_len,
                max_decode_query_len=max_decode_query_len,
                query_start_loc=query_start_loc,
                seq_start_loc=seq_start_loc,
                encoder_seq_lens=encoder_seq_lens,
                encoder_seq_lens_tensor=encoder_seq_lens_tensor,
                encoder_seq_start_loc=encoder_seq_start_loc,
                max_encoder_seq_len=max_encoder_seq_len,
                num_encoder_tokens=num_encoder_tokens,
                cross_slot_mapping=cross_slot_mapping,
                cross_block_tables=cross_block_tables,
                num_prefills=num_prefills,
                num_prefill_tokens=num_prefill_tokens,
                num_decode_tokens=num_decode_tokens,
                slot_mapping=slot_mapping,
                multi_modal_placeholder_index_maps=multi_modal_placeholder_index_maps,
                enable_kv_scales_calculation=enable_kv_scales_calculation,
            )
        else:
            raise NotImplementedError(f"Attention metadata for {self.device} is not implemented yet.")
    
    def build_vllm_config(self) -> VllmConfig:
        return VllmConfig(
            parallel_config=ParallelConfig(
                tensor_parallel_size=self.tp_size,
            ),
            compilation_config=CompilationConfig(),
        )
