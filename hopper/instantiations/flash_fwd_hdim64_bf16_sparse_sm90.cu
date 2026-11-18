// sparse_rope block rule: forward kernel with the block rule compiled in (CollectiveMainloopFwdSm90<..., Sparse=true>).

#include "flash_fwd_launch_template.h"

#ifndef FLASHATTENTION_DISABLE_HDIM64
template void run_mha_fwd_<90, cutlass::bfloat16_t, 64, 64, false, false, false, false, true>(Flash_fwd_params &params, cudaStream_t stream);
#endif
