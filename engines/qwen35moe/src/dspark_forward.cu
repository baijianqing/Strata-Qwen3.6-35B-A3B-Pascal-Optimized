#include "common.hpp"
// GPU assembly of the MiniCPM DSpark 4-draft forward (parity prototype; scratch allocated per call).
#include "dspark_gpu.cuh"

#include <cuda_runtime.h>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace sq {
namespace {

struct DevBuf {
    void* p = nullptr;
    ~DevBuf() { if (p) cudaFree(p); }
    void alloc(size_t bytes) {
        if (bytes && cudaMalloc(&p, bytes) != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(cudaGetLastError()));
    }
    template <typename T> T* ptr() const { return (T*)p; }
};

struct ActBuf {
    DevBuf q, d, s;
    void alloc(int T, int n) {
        q.alloc((size_t)T * n);
        d.alloc((size_t)T * (n / 32) * 4);
        s.alloc((size_t)T * (n / 32) * 4);
    }
    ActQ get(int T, int n) const {
        return ActQ{q.ptr<int8_t>(), d.ptr<float>(), s.ptr<float>()};
    }
};

}  // namespace

bool dspark_gpu_draft(const DSparkW& w, const DQ8& lm_head, int n_vocab,
                      const std::vector<float>& taps_rows, int P,
                      const std::vector<float>& emb_rows, int pos0, int anchor,
                      std::vector<int>& out, std::vector<float>& conf,
                      std::string& err) {
    cudaStream_t s = nullptr;
    const int T = 4, E = w.n_embd, NF = w.n_ff, NH = w.n_head, NKV = w.n_head_kv, HD = w.head_dim;
    const int EH = NH * HD, KV = NKV * HD, TAPS = (int)w.target_layers.size() * E;
    try {
        if ((int)taps_rows.size() != P * TAPS || (int)emb_rows.size() != T * E)
            throw std::runtime_error("DSpark tap/embedding size mismatch");
        DevBuf taps, g, h, hn, att, o, q, k, v, gate, up, ff, logits, xn, ek, ev, bias, r, rq, rqd, rqs, tok, cf;
        ActBuf tap_a, g_a, att_a, xq_a, ff_a;
        taps.alloc((size_t)P * TAPS * 4);
        g.alloc((size_t)P * E * 4);
        ek.alloc((size_t)w.n_layer * P * KV * 4);
        ev.alloc((size_t)w.n_layer * P * KV * 4);
        h.alloc((size_t)T * E * 4);
        hn.alloc((size_t)T * E * 4);
        att.alloc((size_t)T * EH * 4);
        o.alloc((size_t)T * E * 4);
        q.alloc((size_t)T * EH * 4);
        k.alloc((size_t)T * KV * 4);
        v.alloc((size_t)T * KV * 4);
        gate.alloc((size_t)T * NF * 4);
        up.alloc((size_t)T * NF * 4);
        ff.alloc((size_t)T * E * 4);
        logits.alloc((size_t)T * n_vocab * 4);
        xn.alloc((size_t)T * E * 4);
        bias.alloc((size_t)n_vocab * 4);
        r.alloc(256 * 4); rq.alloc(256); rqd.alloc(8 * 4); rqs.alloc(8 * 4);
        tok.alloc((T + 1) * 4); cf.alloc(T * 4);
        tap_a.alloc(P, TAPS);
        g_a.alloc(P, E);
        att_a.alloc(T, EH);
        xq_a.alloc(T, E);
        ff_a.alloc(T, NF);

        const float eps = w.eps;
        auto gemv_chunked = [&](const DQ8& W, ActQ a, int rows, int cols, float* y) {
            for (int i = 0; i < rows; i += 4) {
                const int t = std::min(4, rows - i);
                k_gemv_q8(W, a.row(i, cols), y + (size_t)i * W.rows, t, s);
            }
        };
        auto norm_chunked = [&](float* x, const float* w, float* xf, ActQ aq, int rows, int cols) {
            for (int i = 0; i < rows; i += 4) {
                const int t = std::min(4, rows - i);
                k_rmsnorm_q(x + (size_t)i * cols, w, eps, xf + (size_t)i * cols, aq.row(i, cols), t, s);
            }
        };
        CUDA_CHECK(cudaMemcpyAsync(taps.ptr<float>(), taps_rows.data(), taps_rows.size() * 4, cudaMemcpyHostToDevice, s));
        CUDA_CHECK(cudaMemcpyAsync(h.ptr<float>(), emb_rows.data(), emb_rows.size() * 4, cudaMemcpyHostToDevice, s));
        k_dspark_quant(taps.ptr<float>(), tap_a.get(P, TAPS), P, TAPS, s);
        gemv_chunked(w.fc, tap_a.get(P, TAPS), P, TAPS, g.ptr<float>());
        norm_chunked(g.ptr<float>(), w.enc_norm, g.ptr<float>(), g_a.get(P, E), P, E);
        for (int l = 0; l < w.n_layer; ++l) {
            float* ekl = ek.ptr<float>() + (size_t)l * P * KV;
            float* evl = ev.ptr<float>() + (size_t)l * P * KV;
            gemv_chunked(w.layers[l].k, g_a.get(P, E), P, E, ekl);
            gemv_chunked(w.layers[l].v, g_a.get(P, E), P, E, evl);
            k_dspark_qk_rope(nullptr, ekl, w.layers[l].q_norm, w.layers[l].k_norm, P, NH, NKV, HD, 0, w.eps,
                             false, s);
        }
        // First-layer attention norm from the draft embeddings.
        k_rmsnorm_q(h.ptr<float>(), w.layers[0].attn_norm, w.eps, hn.ptr<float>(), xq_a.get(T, E), T, s);
        for (int l = 0; l < w.n_layer; ++l) {
            const DSparkLayerW& L = w.layers[l];
            k_gemv_q8(L.q, xq_a.get(T, E), q.ptr<float>(), T, s);
            k_gemv_q8(L.k, xq_a.get(T, E), k.ptr<float>(), T, s);
            k_gemv_q8(L.v, xq_a.get(T, E), v.ptr<float>(), T, s);
            k_dspark_qk_rope(q.ptr<float>(), k.ptr<float>(), L.q_norm, L.k_norm, T, NH, NKV, HD, pos0, w.eps,
                             true, s);
            const float* ekl = ek.ptr<float>() + (size_t)l * P * KV;
            const float* evl = ev.ptr<float>() + (size_t)l * P * KV;
            k_dspark_attn(q.ptr<float>(), ekl, evl, k.ptr<float>(), v.ptr<float>(), P, T, NH, NKV, HD,
                          att.ptr<float>(), s);
            k_dspark_quant(att.ptr<float>(), att_a.get(T, EH), T, EH, s);
            k_gemv_q8(L.o, att_a.get(T, EH), o.ptr<float>(), T, s);
            k_add_rmsnorm_q(h.ptr<float>(), o.ptr<float>(), L.ffn_norm, w.eps, hn.ptr<float>(), xq_a.get(T, E), T, s);
            k_gemv_q8(L.gate, xq_a.get(T, E), gate.ptr<float>(), T, s);
            k_gemv_q8(L.up, xq_a.get(T, E), up.ptr<float>(), T, s);
            k_dspark_swiglu(gate.ptr<float>(), up.ptr<float>(), ff_a.get(T, NF), T, NF, s);
            k_gemv_q8(L.down, ff_a.get(T, NF), ff.ptr<float>(), T, s);
            if (l + 1 < w.n_layer) {
                k_add_rmsnorm_q(h.ptr<float>(), ff.ptr<float>(), w.layers[l + 1].attn_norm, w.eps,
                                hn.ptr<float>(), xq_a.get(T, E), T, s);
            } else {
                k_add_rmsnorm_q(h.ptr<float>(), ff.ptr<float>(), w.out_norm, w.eps, xn.ptr<float>(),
                                xq_a.get(T, E), T, s);
            }
        }
        k_gemv_q8(lm_head, xq_a.get(T, E), logits.ptr<float>(), T, s);
        k_dspark_markov_step(w.markov_w1, w.markov_w2, anchor, T, logits.ptr<float>(), xn.ptr<float>(),
                             n_vocab, E, w.conf_proj, w.conf_bias, r.ptr<float>(), rq.ptr<int8_t>(),
                             rqd.ptr<float>(), rqs.ptr<float>(), bias.ptr<float>(), tok.ptr<int>(),
                             cf.ptr<float>(), s);
        CUDA_CHECK(cudaStreamSynchronize(s));
        out.resize(T); conf.resize(T);
        CUDA_CHECK(cudaMemcpy(out.data(), tok.ptr<int>() + 1, T * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(conf.data(), cf.ptr<float>(), T * 4, cudaMemcpyDeviceToHost));
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

}  // namespace sq
