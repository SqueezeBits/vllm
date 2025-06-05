This directory contains scripts for micro benchmarks of vllm models. It currently targets `Qwen/Qwen3-30B-A3B` model, and available modules to test are as follows:

- Module Level (with user-defined number of layers)
    - Qwen3DecoderLayer
    - Qwen3MoeSparseMoeBlock
    - Qwen3MoeAttention
- Op Level
    - fused_experts kernel
    - topk_softmax kernel
    - flash_attention kernel
    - RMSNorm
    - RotaryEmbedding

Each modules containing MoE block(Qwen3DecoderLayer, Qwen3MoeSparseMoeBlock, fused_experts) can be tested with EP and DP. And pre-trained model weights will be loaded for module level tests.

### Prerequisites
The original vllm should be installed, following their original documentations.

### How to Run
- Without EP

    For module level tests, where forward context(attention metadata) is required for execution, using V1 engine is not implemented yet. 
    ```
    VLLM_USE_V1=0 python qwen3_decoder_layer.py
    VLLM_USE_V1=0 python qwen3_decoder_layer.py --num-layers 2
    python fused_experts.py
    ```
    Random inputs will be used by default, but it's also available to save(`--save-inputs`) and load(`--load-inputs`) inputs for reproducibility in every script.

- With EP

    By default, TP is applied to Attention module with TP size having the same value as EP size. To disable it, set `DISABLE_ATTN_TP=1`. For EP only(without DP), it's recommended to save and load inputs to ensure each module on different device receive the same inputs.
    ```
    DISABLE_ATTN_TP=1 VLLM_USE_V1=0 python qwen3_decoder_layer.py --ep-size 2
    VLLM_USE_V1=0 python qwen3_moe_sparse_moe_block.py --ep-size 2
    python fused_experts.py --ep-size 2
    ```

- With EP + DP

    If DP is enabled with `--enable-data-parallel`, DP size is set to the same value as EP size. Note that communications for DP (broadcast and all_reduce) are performed outside of the fused_experts kernel, so it's no use to test DP on fused_experts kernel.
    ```
    DISABLE_ATTN_TP=1 VLLM_USE_V1=0 python qwen3_decoder_layer.py --ep-size 2 --enable-data-parallel
    VLLM_USE_V1=0 python qwen3_moe_sparse_moe_block.py --ep-size 2 --enable-data-parallel
    ```