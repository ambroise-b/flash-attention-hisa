/******************************************************************************
 * sparse_rope block rule, decode: K / V loads from per-chunk ring buffers (cp.async), dropped chunks not read.
 ******************************************************************************/

#pragma once

#include <cute/tensor.hpp>

#include "utils.h"

namespace flash {

using namespace cute;

// Same interface as PagedKVManager (the PagedKVNonTMA path of the mainloop), for the sparse_rope decode kernel
// (CollectiveMainloopFwdSm90<..., PagedKVNonTMA = true, Sparse = true>).
//
// K is stored as 4 ring buffers, one per 16-feature chunk of the chunked layout (chunk c = RoPE pairs 8c .. 8c + 7):
// Kc[c] of shape (b, R_c, h_k, 16), contiguous. V is one ring (b, R_V, h_k, d_v) (params.ptr_V, stride_V). The key at
// position j lives in slot j % R; every R is a multiple of kBlockN, so the rows of key block n_block are the slots
// (n_block * kBlockN) % R + r, r < kBlockN (no wrap inside a block). Key block n_block keeps the chunks >= K0, where
// K0 follows from the block-rule distance nd = bi - n_block (kBlockN = rule block) and the table header nd_1 .. nd_4
// (see Flash_fwd_params::sparse_a_ptr); chunks below K0 are not read (cp.async with zero source size, i.e. zero-fill
// without a memory access; the QK^T GEMM skips them anyway). K0 = 4: no pair kept, nothing of K is read.
// With 8 threads per 64-feature row and 8 elements per 16-byte load, each thread always loads the same chunk.
template <int kBlockN, int kHeadDim, int kHeadDimV, int NumThreads, typename Element>
struct SparseKVManager {
    static_assert(kHeadDim == 64 && kHeadDimV == 64, "sparse_rope decode: head_dim 64");
    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);   // 8
    static constexpr int kGmemThreadsPerRow = kHeadDim / kGmemElemsPerLoad;                // 8
    static_assert(NumThreads % kGmemThreadsPerRow == 0);
    using GmemCopyAtomCpAsync = cute::Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<uint128_t>, Element>;
    using GmemLayoutAtomKVCpAsync = Layout<Shape <Int<NumThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                           Stride<Int<kGmemThreadsPerRow>, _1>>;
    using GmemTiledCopyKVCpAsync = decltype(
        make_tiled_copy(GmemCopyAtomCpAsync{}, GmemLayoutAtomKVCpAsync{}, Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));
    using GmemThrCopyKVCpAsync = decltype(GmemTiledCopyKVCpAsync{}.get_thread_slice(int(0)));

    GmemTiledCopyKVCpAsync gmem_tiled_copy_kv;
    GmemThrCopyKVCpAsync const gmem_thr_copy_kv;
    int const seqlen_k;
    int const bi;               // rule block of the query (position seqlen_k - 1)
    int nd1, nd2, nd3, nd4;
    Element const* k_ptr;       // this thread's chunk, batch, kv head and column (slot 0)
    int k_rows;                 // R of this thread's chunk
    int64_t const k_row_stride; // h_k * 16
    int chunk;                  // this thread's chunk
    Element const* v_ptr;       // batch, kv head, column (slot 0)
    int const v_rows;
    int64_t const v_row_stride;

    CUTLASS_DEVICE
    SparseKVManager(Element const* const* kc, int const* kc_rows, int const h_k,
                    Element const* ptr_V, int const v_rows_, int64_t const v_row_stride_, int64_t const v_batch_stride,
                    int const* sparse_a, int const rule_block,
                    int const bidb, int const bidh_kv, int const thread_idx, int const seqlen_k_)
        : gmem_thr_copy_kv(gmem_tiled_copy_kv.get_thread_slice(thread_idx))
        , seqlen_k(seqlen_k_)
        , bi((seqlen_k_ - 1) / rule_block)
        , k_row_stride(int64_t(h_k) * 16)
        , v_rows(v_rows_)
        , v_row_stride(v_row_stride_)
    {
        nd1 = __ldg(sparse_a); nd2 = __ldg(sparse_a + 1); nd3 = __ldg(sparse_a + 2); nd4 = __ldg(sparse_a + 3);
        int const col = (thread_idx % kGmemThreadsPerRow) * kGmemElemsPerLoad;  // this thread's 8 features
        chunk = col / 16;
        Element const* kc_c = chunk == 0 ? kc[0] : chunk == 1 ? kc[1] : chunk == 2 ? kc[2] : kc[3];
        k_rows = chunk == 0 ? kc_rows[0] : chunk == 1 ? kc_rows[1] : chunk == 2 ? kc_rows[2] : kc_rows[3];
        k_ptr = kc_c + int64_t(bidb) * k_rows * k_row_stride + int64_t(bidh_kv) * 16 + (col % 16);
        v_ptr = ptr_V + int64_t(bidb) * v_batch_stride + int64_t(bidh_kv) * kHeadDimV + col;
    };

    CUTLASS_DEVICE int k0(int const n_block) const {
        int const nd = max(bi - n_block, 0);
        return int(nd >= nd1) + int(nd >= nd2) + int(nd >= nd3) + int(nd >= nd4);
    }

    template <bool Seqlenk_mask=false, bool First_iter=false>
    CUTLASS_DEVICE void load_page_table(const int n_block) {}

    template <bool Seqlenk_mask=false, typename TensorK>
    CUTLASS_DEVICE void load_K(const int n_block, TensorK &&sK) {
        Tensor tKsK = gmem_thr_copy_kv.partition_D(sK);
        Tensor cK = cute::make_identity_tensor(Shape<Int<kBlockN>, Int<kHeadDim>>{});
        Tensor tKcK = gmem_thr_copy_kv.partition_S(cK);
        static_assert(CUTE_STATIC_V(size<2>(tKsK)) == 1, "one 16-byte load per row and thread");
        bool const keep = chunk >= k0(n_block);
        int const slot0 = (n_block * kBlockN) % k_rows;
        int const row_limit = seqlen_k - n_block * kBlockN;
        #pragma unroll
        for (int m = 0; m < size<1>(tKsK); ++m) {
            int const row = get<0>(tKcK(_0{}, m, _0{}));
            bool const should_load = keep && (!Seqlenk_mask || row < row_limit);
            Tensor src = make_tensor(make_gmem_ptr(k_ptr + (slot0 + row) * k_row_stride), Shape<Int<kGmemElemsPerLoad>>{});
            cute::copy(gmem_tiled_copy_kv.with(should_load), src, tKsK(_, m, _0{}));
        }
    };

    template <bool Seqlenk_mask=false, typename TensorV>
    CUTLASS_DEVICE void load_V(const int n_block, TensorV &&sV) {
        Tensor tVsV = gmem_thr_copy_kv.partition_D(sV);
        Tensor cV = cute::make_identity_tensor(Shape<Int<kBlockN>, Int<kHeadDimV>>{});
        Tensor tVcV = gmem_thr_copy_kv.partition_S(cV);
        static_assert(CUTE_STATIC_V(size<2>(tVsV)) == 1, "one 16-byte load per row and thread");
        int const slot0 = (n_block * kBlockN) % v_rows;
        int const row_limit = seqlen_k - n_block * kBlockN;
        #pragma unroll
        for (int m = 0; m < size<1>(tVsV); ++m) {
            int const row = get<0>(tVcV(_0{}, m, _0{}));
            bool const should_load = !Seqlenk_mask || row < row_limit;  // zero-fill out of bounds (P = 0 there)
            Tensor src = make_tensor(make_gmem_ptr(v_ptr + (slot0 + row) * v_row_stride), Shape<Int<kGmemElemsPerLoad>>{});
            cute::copy(gmem_tiled_copy_kv.with(should_load), src, tVsV(_, m, _0{}));
        }
    };
};

} // namespace flash
