#include "common.hpp"
// MiniCPM DSpark drafter kernels: 8-head Q/K norm + full-256 NeoX RoPE, non-causal block attention,
// F32->ActQ quantization and the 6144-wide SwiGLU.  The dense matmuls reuse k_gemv_q8.
#include "dspark_gpu.cuh"
#include <cuda_fp16.h>

namespace sq {

__global__ void qk_rope_kernel(float* q, float* k, const float* qn, const float* kn, int T, int nh, int nkv,
                               int hd, int pos0, float eps, bool with_q) {
    const int t = blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
    const int hh = with_q ? h : h + nh;
    const bool is_q = hh < nh;
    float* v = is_q ? q + ((size_t)t * nh + hh) * hd : k + ((size_t)t * nkv + (hh - nh)) * hd;
    const float* w = is_q ? qn : kn;
    __shared__ float sh[256];
    float x = v[tid];
    sh[tid] = x * x;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) sh[tid] += sh[tid + stride];
        __syncthreads();
    }
    x = x * rsqrtf(sh[0] / hd + eps) * w[tid];
    sh[tid] = x;
    __syncthreads();
    if (tid < hd / 2) {
        const float a = sh[tid], b = sh[tid + hd / 2];
        const float ang = (pos0 + t) * powf(1e7f, -2.0f * tid / hd);
        const float c = cosf(ang), s = sinf(ang);
        v[tid] = a * c - b * s;
        v[tid + hd / 2] = a * s + b * c;
    }
}

void k_dspark_qk_rope(float* q, float* k, const float* qn, const float* kn, int T, int nh, int nkv, int hd,
                      int pos0, float eps, bool with_q, cudaStream_t s) {
    dim3 grid(T, with_q ? nh + nkv : nkv);
    qk_rope_kernel<<<grid, hd, 0, s>>>(q, k, qn, kn, T, nh, nkv, hd, pos0, eps, with_q);
}

// One thread per (token, head): serial over P+T keys.  Correct first, then optimize.
__global__ void attn_kernel(const float* q, const float* ek, const float* ev, const float* bk, const float* bv,
                            int P, int T, int nh, int nkv, int hd, float* out) {
    const int t = blockIdx.x, h = blockIdx.y;
    const int kv = h / (nh / nkv);
    const int NK = P + T;
    const float* qh = q + ((size_t)t * nh + h) * hd;
    float* oh = out + ((size_t)t * nh + h) * hd;
    for (int d = 0; d < hd; ++d) oh[d] = 0;
    float mx = -INFINITY;
    extern __shared__ float scores[];
    for (int j = 0; j < NK; ++j) {
        const float* kk = j < P ? ek + ((size_t)j * nkv + kv) * hd : bk + ((size_t)(j - P) * nkv + kv) * hd;
        float s = 0;
        for (int d = 0; d < hd; ++d) s += qh[d] * kk[d];
        scores[j] = s / 16.0f;
        mx = fmaxf(mx, scores[j]);
    }
    float sum = 0;
    for (int j = 0; j < NK; ++j) { scores[j] = expf(scores[j] - mx); sum += scores[j]; }
    for (int j = 0; j < NK; ++j) {
        const float p = scores[j] / sum;
        const float* vv = j < P ? ev + ((size_t)j * nkv + kv) * hd : bv + ((size_t)(j - P) * nkv + kv) * hd;
        for (int d = 0; d < hd; ++d) oh[d] += p * vv[d];
    }
}

void k_dspark_attn(const float* q, const float* ek, const float* ev, const float* bk, const float* bv,
                   int P, int T, int nh, int nkv, int hd, float* out, cudaStream_t s) {
    dim3 grid(T, nh);
    attn_kernel<<<grid, 1, (size_t)(P + T) * sizeof(float), s>>>(q, ek, ev, bk, bv, P, T, nh, nkv, hd, out);
}

__global__ void quant_kernel(const float* x, ActQ out, int n) {
    const int row = blockIdx.x;
    const int b = threadIdx.x;                // one thread per 32-value block
    const int base = b * 32;
    const float* src = x + (size_t)row * n + base;
    int8_t* q = out.q + (size_t)row * n + base;
    float mx = 0;
    for (int i = 0; i < 32; ++i) mx = fmaxf(mx, fabsf(src[i]));
    const float d = mx / 127.0f;
    int sum = 0;
    for (int i = 0; i < 32; ++i) {
        const int v = d > 0 ? (int)lrintf(src[i] / d) : 0;
        q[i] = (int8_t)v;
        sum += v;
    }
    out.d[(size_t)row * (n / 32) + b] = d;
    out.s[(size_t)row * (n / 32) + b] = d * sum;
}

void k_dspark_quant(const float* x, ActQ out, int T, int n, cudaStream_t s) {
    quant_kernel<<<T, n / 32, 0, s>>>(x, out, n);
}

__global__ void swiglu_kernel(const float* gate, const float* up, ActQ out, int nf) {
    const int row = blockIdx.x;
    const int b = threadIdx.x;
    const int base = b * 32;
    float act[32];
    float mx = 0;
    for (int i = 0; i < 32; ++i) {
        const float g = gate[(size_t)row * nf + base + i];
        act[i] = (g / (1.0f + expf(-g))) * up[(size_t)row * nf + base + i];
        mx = fmaxf(mx, fabsf(act[i]));
    }
    const float d = mx / 127.0f;
    int sum = 0;
    for (int i = 0; i < 32; ++i) {
        const int v = d > 0 ? (int)lrintf(act[i] / d) : 0;
        out.q[(size_t)row * nf + base + i] = (int8_t)v;
        sum += v;
    }
    out.d[(size_t)row * (nf / 32) + b] = d;
    out.s[(size_t)row * (nf / 32) + b] = d * sum;
}


__global__ void markov_row_kernel(const int8_t* qs, const uint16_t* ds, const int* prev_row, int cols,
                                  float* r, int8_t* rq, float* rqd, float* rqs) {
    const int row = *prev_row;
    const int b = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int i = b * 32 + lane;
    const float scale = __half2float(*reinterpret_cast<const __half*>(&ds[(size_t)row * (cols / 32) + b]));
    const float v = qs[(size_t)row * cols + i] * scale;
    r[i] = v;
    float mx = fabsf(v);
    for (int off = 16; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_down_sync(0xffffffff, mx, off));
    mx = __shfl_sync(0xffffffff, mx, 0);
    const float d = mx / 127.0f;
    const int qi = d > 0 ? (int)lrintf(v / d) : 0;
    rq[i] = (int8_t)qi;
    int sum = qi;
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffff, sum, off);
    if (lane == 0) { rqd[b] = d; rqs[b] = d * sum; }
}

__global__ void bias_argmax_kernel(const float* logits, const float* bias, int V, int* token) {
    float best = -INFINITY; int bi = 0;
    for (int i = threadIdx.x; i < V; i += blockDim.x) {
        const float v = logits[i] + bias[i];
        if (v > best) { best = v; bi = i; }
    }
    __shared__ float sv[256]; __shared__ int si[256];
    sv[threadIdx.x] = best; si[threadIdx.x] = bi; __syncthreads();
    for (int off = 128; off > 0; off >>= 1) {
        if (threadIdx.x < off && sv[threadIdx.x + off] > sv[threadIdx.x]) {
            sv[threadIdx.x] = sv[threadIdx.x + off]; si[threadIdx.x] = si[threadIdx.x + off];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) *token = si[0];
}

__global__ void conf_kernel(const float* x, const float* r, const float* cw, const float* cb,
                            int E, float* conf) {
    float s = 0.0f;
    for (int i = threadIdx.x; i < E; i += blockDim.x) s += x[i] * cw[i];
    if (threadIdx.x < 256) s += r[threadIdx.x] * cw[E + threadIdx.x];
    __shared__ float red[256]; red[threadIdx.x] = s; __syncthreads();
    for (int off = 128; off > 0; off >>= 1) {
        if (threadIdx.x < off) red[threadIdx.x] += red[threadIdx.x + off];
        __syncthreads();
    }
    if (threadIdx.x == 0) *conf = 1.0f / (1.0f + expf(-(red[0] + cb[0])));
}


void k_dspark_markov_step(const DQ8& w1, const DQ8& w2, int anchor, int T,
                          const float* logits, const float* xn, int V, int E,
                          const float* cw, const float* cb,
                          float* r, int8_t* rq, float* rqd, float* rqs, float* bias,
                          int* tok, float* conf, cudaStream_t s) {
    CUDA_CHECK(cudaMemcpyAsync(tok, &anchor, 4, cudaMemcpyHostToDevice, s));
    ActQ rqa{rq, rqd, rqs};
    for (int t = 0; t < T; ++t) {
        markov_row_kernel<<<1, 256, 0, s>>>(w1.qs, w1.d, tok + t, 256, r, rq, rqd, rqs);
        k_gemv_q8(w2, rqa, bias, 1, s);
        bias_argmax_kernel<<<1, 256, 0, s>>>(logits + (size_t)t * V, bias, V, tok + t + 1);
        conf_kernel<<<1, 256, 0, s>>>(xn + (size_t)t * E, r, cw, cb, E, conf + t);
    }
}

void k_dspark_swiglu(const float* gate, const float* up, ActQ out, int T, int nf, cudaStream_t s) {
    swiglu_kernel<<<T, nf / 32, 0, s>>>(gate, up, out, nf);
}

}  // namespace sq
