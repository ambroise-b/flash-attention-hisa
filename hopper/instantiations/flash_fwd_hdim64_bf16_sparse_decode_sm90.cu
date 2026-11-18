// sparse_rope block rule, decode: seqlen_q = 1, q heads packed (PackGQA), K / V from per-chunk ring buffers
// (PagedKVNonTMA path with SparseKVManager, sparse_kv.h); with and without split-KV.

#include "flash_fwd_launch_template.h"

#ifndef FLASHATTENTION_DISABLE_HDIM64
template void run_mha_fwd_<90, cutlass::bfloat16_t, 64, 64, true, true, false, true, true>(Flash_fwd_params &params, cudaStream_t stream);
template void run_mha_fwd_<90, cutlass::bfloat16_t, 64, 64, false, true, false, true, true>(Flash_fwd_params &params, cudaStream_t stream);
#endif
