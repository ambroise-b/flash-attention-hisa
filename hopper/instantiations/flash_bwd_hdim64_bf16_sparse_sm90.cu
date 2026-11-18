// sparse_rope block rule: backward kernel with the block rule compiled in (CollectiveMainloopBwdSm90<..., Sparse=true>),
// same tile configuration as run_mha_bwd_hdim64 (sm90, no softcap).

#include "flash_bwd_launch_template.h"

#ifndef FLASHATTENTION_DISABLE_HDIM64
void run_mha_bwd_sparse_hdim64_bf16_sm90(Flash_bwd_params &params, cudaStream_t stream) {
    CAUSAL_LOCAL_SWITCH(params.is_causal, params.is_local, Is_causal, Is_local, [&] {
        if constexpr (Is_causal || Is_local) {
            run_mha_bwd_dispatch<90, cutlass::bfloat16_t, 128, 128, 64, Is_causal, Is_local, false, 2, 2, true, false, false, 2, 1, 2, 2, false, true>(params, stream);
        } else {
            throw std::runtime_error("sparse_rope block rule: causal attention only");
        }
    });
}
#endif
