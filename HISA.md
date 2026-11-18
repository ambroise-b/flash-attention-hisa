# HiSA fork of FlashAttention-3

This branch (`hisa`) adds **Flash-HiSA**, the block-rule kernels of HiSA (Hidden Sparse Attention), to the Hopper
(sm90) kernels of FlashAttention-3, in a single commit on top of upstream commit `bb25ba3c` (version 3.0.1).
The Python package, documentation and tests are in the main repository:
**https://github.com/ambroise-b/hisa** (this fork is its `third_party/flash-attention` submodule).

HiSA keeps the score part of RoPE frequency pair l only while its rotation (i - j) * theta_l stays below an angle
budget. Flash-HiSA applies this per (query block, key block) of 128 tokens at the blocks' minimum distance, in
16-feature chunks (8 pairs): chunks of dropped pairs are skipped in Q K^T, and the dropped pairs of the boundary
chunk are zeroed in shared-memory Q. The new kernels are separate compile-time instantiations (head dim 64, bf16,
sm90); the dense kernels are unchanged.

| File | Change |
|---|---|
| `hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp` | forward: chunk-sliced Q K^T, per-K0 straight-line GMMA segments, Q zeroing, blocks beyond the reach masked |
| `hopper/mainloop_bwd_sm90_tma_gmma_ws.hpp` | backward: same rule, S^T skips dropped chunks, dQ columns of dropped pairs masked |
| `hopper/sparse_kv.h` | decoding: K from 4 per-chunk ring buffers and V from a ring with cp.async; dropped chunks never read |
| `hopper/flash_api_stable.cpp`, `flash.h` | op arguments `sparse_a` (block-rule table), `sparse_rule_block`, `sparse_kc0..3` (ring buffers) |
| `hopper/flash_{fwd,bwd}_launch_template.h`, `tile_size.h`, `utils.h` | dispatch of the Sparse variants, 64-row decode tiles, helpers |
| `hopper/instantiations/flash_*_hdim64_bf16_sparse*_sm90.cu`, `setup.py` | the new instantiations and their build |

Build (sm90, hdim 64, bf16 only): see `scripts/build_flash.sh` in the main repository. `FA3_EXTRA_NVCC_FLAGS`
passes extra defines; the `SPARSE_ROPE_EXP_*` switches are experiments, off by default.
