#pragma once
#include "kernels.cuh"

namespace sq {

void k_dspark_qk_rope(float* q, float* k, const float* qn, const float* kn, int T, int n_head, int n_head_kv,
                      int hd, int pos0, float eps, bool with_q, cudaStream_t s);
void k_dspark_attn(const float* q, const float* ek, const float* ev, const float* bk, const float* bv,
                   int P, int T, int n_head, int n_head_kv, int hd, float* out, cudaStream_t s);
void k_dspark_quant(const float* x, ActQ out, int T, int n, cudaStream_t s);
void k_dspark_swiglu(const float* gate, const float* up, ActQ out, int T, int nf, cudaStream_t s);
void k_dspark_markov_step(const DQ8& w1, const DQ8& w2, int anchor, int T,
                          const float* logits, const float* xn, int V, int E,
                          const float* cw, const float* cb,
                          float* r, int8_t* rq, float* rqd, float* rqs, float* bias,
                          int* tok, float* conf, cudaStream_t s);

bool dspark_gpu_draft(const DSparkW& w, const DQ8& lm_head, int n_vocab,
                      const std::vector<float>& taps_rows, int P,
                      const std::vector<float>& emb_rows, int pos0, int anchor,
                      std::vector<int>& out, std::vector<float>& conf,
                      std::string& err);

}  // namespace sq
