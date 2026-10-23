// cpu_moe.cpp - the CPU half of the MoE: kernel dispatch and the worker pool.  See cpu_moe.hpp.
//
// The kernels themselves are in cpu_avx512.cpp and cpu_avx2.cpp, each compiled for its instruction set; this file is
// compiled for the baseline (x86-64) and picks one set at run time.
#include "cpu_moe.hpp"

#include "common.hpp"
#include "cpu_iq.hpp"
#include "cpu_iq3_cache.hpp"
#include "cpu_kernels.hpp"
#include "native_iq/iq_avx2.hpp"

#include <immintrin.h>

#include <chrono>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>

#ifdef _WIN32
#include <intrin.h>
#include <windows.h>
#else
#include <cpuid.h>
#include <pthread.h>
#include <sched.h>
#include <time.h>
#endif

namespace sq {

// ================================================================== instruction set
namespace {

uint64_t phase_now_ns() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

/// Scheduled CPU time of the calling worker (not wall time), so preemption is separable from compute.
uint64_t thread_cpu_now_ns() {
#ifndef _WIN32
    timespec ts;
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts) == 0)
        return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
#endif
    return 0;
}

// Profile=false is an empty scope: the ordinary kernel has no clock calls or counter stores.
template<bool Profile> struct PhaseSpan {
    uint64_t* target = nullptr;
    uint64_t start = 0;
    explicit PhaseSpan(uint64_t* counter) {
        if constexpr (Profile) { target = counter; start = phase_now_ns(); }
    }
    ~PhaseSpan() { if constexpr (Profile) *target += phase_now_ns() - start; }
};

uint64_t phase_env_count(const char* name, uint64_t fallback, bool allow_zero) {
    const char* value = std::getenv(name);
    if (!value || !*value) return fallback;
    char* end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    SQ_CHECK(value[0] >= '0' && value[0] <= '9' && end && *end == '\0' && errno != ERANGE &&
             parsed <= 1000000 && (allow_zero || parsed > 0), "%s must be an integer in %d..1000000", name, allow_zero ? 0 : 1);
    return parsed;
}

void cpuid(int leaf, int sub, unsigned r[4]) {
#ifdef _WIN32
    int x[4];
    __cpuidex(x, leaf, sub);
    for (int i = 0; i < 4; ++i) r[i] = (unsigned)x[i];
#else
    __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
#endif
}

unsigned long long xgetbv0() {
#ifdef _WIN32
    return _xgetbv(0);
#else
    unsigned lo, hi;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    return ((unsigned long long)hi << 32) | lo;
#endif
}

struct Features { bool avx2 = false, avx512 = false; };

Features detect() {
    Features f;
    unsigned r[4];
    cpuid(0, 0, r);
    if (r[0] < 7) return f;
    cpuid(1, 0, r);
    const bool osxsave = r[2] & (1u << 27), fma = r[2] & (1u << 12), f16c = r[2] & (1u << 29);
    if (!osxsave) return f;
    const unsigned long long xcr0 = xgetbv0();
    const bool ymm = (xcr0 & 0x6) == 0x6, zmm = (xcr0 & 0xE6) == 0xE6;
    cpuid(7, 0, r);
    f.avx2 = ymm && fma && f16c && (r[1] & (1u << 5));
    // F, DQ, BW, VL and VNNI: what cpu_avx512.cpp is compiled for
    const unsigned ebx_need = (1u << 16) | (1u << 17) | (1u << 30) | (1u << 31);
    f.avx512 = f.avx2 && zmm && (r[1] & ebx_need) == ebx_need && (r[2] & (1u << 11));
    return f;
}

const CpuKernels kAvx512{"AVX-512 VNNI", avx512::quantize_q8k, avx512::dot_q4k, avx512::dot_q5k, avx512::dot_q6k};
const CpuKernels kAvx2{"AVX2", avx2::quantize_q8k, avx2::dot_q4k, avx2::dot_q5k, avx2::dot_q6k};
const CpuKernels* g_kernels = nullptr;

bool k23_reuse_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("STRATA_CPU_K23_REUSE");
        return value && std::atoi(value) != 0;
    }();
    return enabled;
}

const CpuKernels& kernels() {
    if (!g_kernels) {
        const Features f = detect();
        SQ_CHECK(f.avx2, "this CPU has no AVX2 (with FMA and F16C): the expert kernels need it");
        const char* env = std::getenv("STRATA_CPU_ISA");
        g_kernels = f.avx512 && !(env && std::string(env) == "avx2") ? &kAvx512 : &kAvx2;
    }
    return *g_kernels;
}

}  // namespace

const char* cpu_isa() { return kernels().name; }

bool cpu_has_avx2() { return detect().avx2; }

bool cpu_force_isa(const char* name) {
    const Features f = detect();
    const std::string n(name);
    if (n == "avx512" && f.avx512) { g_kernels = &kAvx512; return true; }
    if (n == "avx2" && f.avx2) { g_kernels = &kAvx2; return true; }
    return false;
}

void quantize_q8k(const float* x, Q8K* y, int n) { kernels().quantize_q8k(x, y, n); }

DotFn dot_fn(uint32_t type) {
    const CpuKernels& k = kernels();
    switch (type) {
        case T_Q4_K: return k.dot_q4k;
        case T_Q5_K: return k.dot_q5k;
        case T_Q6_K: return k.dot_q6k;
        case T_Q2_K: return native_iq::dot_q2k;
        case T_Q3_K: return native_iq::dot_q3k;
        case T_IQ2_S: return native_iq::dot_iq2s;
        case T_IQ3_S: return native_iq::dot_iq3s;
        case T_IQ4_XS: return native_iq::dot_iq4xs;
        default: return nullptr;
    }
}

float dot_row(uint32_t type, const void* w, const Q8K* a, int nb) {
    const DotFn f = dot_fn(type);
    return f ? f(w, a, nb) : dot_row_ref(type, w, a, nb);
}

float dot_row_ref(uint32_t type, const void* w, const Q8K* a, int nb) {
    std::vector<float> wf((size_t)nb * 256);
    dequant_row(type, w, wf.data(), (int64_t)nb * 256);
    double s = 0;
    for (int i = 0; i < nb; ++i)
        for (int k = 0; k < 256; ++k) s += (double)wf[(size_t)i * 256 + k] * a[i].nat[k] * a[i].d;
    return (float)s;
}

// ================================================================== worker pool
CpuMoe::CpuMoe() = default;
CpuMoe::~CpuMoe() { stop(); }
size_t CpuMoe::iq3_down_cache_bytes() const { return iq3_down_cache_ ? iq3_down_cache_->bytes() : 0; }
uint64_t CpuMoe::iq3_down_cache_rows() const {
    if (!iq3_down_cache_) return 0;
    uint64_t result = 0;
    for (const auto& scratch : scratch_) result += scratch.nibble_rows;
    return result;
}

void CpuMoe::start(int n_threads, int n_layer, const std::vector<ExpertLayerDesc>& layers, Mailbox* mb, Result* res,
                   bool pin_threads) {
    SQ_CHECK(threads_.empty(), "CPU worker pool must be stopped before start");
    n_threads_ = n_threads;
    n_layer_ = n_layer;
    worker_spins_ = 200000;
    if (const char* value = std::getenv("STRATA_CPU_WORKER_SPINS"); value && *value) {
        char* end = nullptr;
        const long parsed = std::strtol(value, &end, 10);
        SQ_CHECK(value[0] >= '0' && value[0] <= '9' && end && *end == '\0' && parsed >= 0 && parsed <= 1000000,
                 "STRATA_CPU_WORKER_SPINS must be an integer in 0..1000000");
        worker_spins_ = (int)parsed;
    }
    log("CPU worker idle spin budget: %d iterations", worker_spins_);
    completion_spins_ = -1;
    if (const char* value = std::getenv("STRATA_CPU_COMPLETION_SPINS"); value && *value) {
        char* end = nullptr;
        errno = 0;
        const long parsed = std::strtol(value, &end, 10);
        SQ_CHECK(end != value && end && *end == '\0' && errno != ERANGE && parsed <= 1000000,
                 "STRATA_CPU_COMPLETION_SPINS must be an integer <= 1000000; negative selects legacy waiting");
        completion_spins_ = parsed < 0 ? -1 : (int)parsed;
    }
    if (completion_spins_ >= 0)
        log("CPU completion wait: at most %d pauses, then atomic wait", completion_spins_);
    prompt_completion_spins_ = completion_spins_;
    if (const char* value = std::getenv("STRATA_PROMPT_COMPLETION_SPINS"); value && *value) {
        char* end = nullptr;
        errno = 0;
        const long parsed = std::strtol(value, &end, 10);
        SQ_CHECK(value[0] >= '0' && value[0] <= '9' && end && *end == '\0' && errno != ERANGE &&
                 parsed >= 0 && parsed <= 1000000, "STRATA_PROMPT_COMPLETION_SPINS must be an integer in 0..1000000");
        prompt_completion_spins_ = (int)parsed;
        log("CPU prompt completion wait: at most %d pauses, then atomic wait", prompt_completion_spins_);
    }
    // Workers never read a mutable current-job wait mode. A previous job's final notification may arrive
    // after the host starts another job; notifying whenever either immutable mode can sleep is harmless.
    notify_completion_ = completion_spins_ >= 0 || prompt_completion_spins_ >= 0;
    phase_enabled_ = false;
    if (const char* value = std::getenv("STRATA_CPU_PHASE_PROFILE"); value && *value) {
        SQ_CHECK(std::strcmp(value, "0") == 0 || std::strcmp(value, "1") == 0,
                 "STRATA_CPU_PHASE_PROFILE must be 0 or 1");
        phase_enabled_ = std::strcmp(value, "1") == 0;
    }
    phase_active_ = phase_logged_ = false;
    phase_layer_ = -1;
    phase_profile_ = {};
    phase_workers_.reset();
    if (phase_trace_fp_) { std::fclose(phase_trace_fp_); phase_trace_fp_ = nullptr; }
    if (phase_workers_fp_) { std::fclose(phase_workers_fp_); phase_workers_fp_ = nullptr; }
    const char* phase_trace_path = std::getenv("STRATA_CPU_PHASE_TRACE");
    const char* phase_workers_path = std::getenv("STRATA_CPU_PHASE_WORKERS");
    if ((phase_trace_path && *phase_trace_path) || (phase_workers_path && *phase_workers_path))
        phase_enabled_ = true;
    if (phase_enabled_) {
        phase_skip_ = phase_env_count("STRATA_CPU_PHASE_PROFILE_SKIP", 40, true);
        phase_limit_ = phase_env_count("STRATA_CPU_PHASE_PROFILE_JOBS", 400, false);
        phase_workers_ = std::make_unique<PhaseWorker[]>(n_threads);
        if (phase_trace_path && *phase_trace_path) {
            phase_trace_fp_ = std::fopen(phase_trace_path, "w");
            SQ_CHECK(phase_trace_fp_, "cannot open STRATA_CPU_PHASE_TRACE %s", phase_trace_path);
            std::fprintf(phase_trace_fp_, "job_index,dispatch_mono_ns,kind,layer,n_tok,n_miss,pairs,"
                                          "dispatch_to_first_start_ns,start_spread_ns,end_min_ns,"
                                          "dispatch_to_last_end_ns,join_wall_ns,workers_cpu_ns,"
                                          "cpu_max_ns,cpu_min_ns,max_worker_elapsed_ns\n");
            std::fflush(phase_trace_fp_);
        }
        if (phase_workers_path && *phase_workers_path) {
            phase_workers_fp_ = std::fopen(phase_workers_path, "w");
            SQ_CHECK(phase_workers_fp_, "cannot open STRATA_CPU_PHASE_WORKERS %s", phase_workers_path);
            std::fprintf(phase_workers_fp_, "job_index,kind,layer,n_tok,tid,start_offset_ns,end_offset_ns,cpu_ns\n");
            std::fflush(phase_workers_fp_);
        }
        log("CPU phase diagnostics: event-layer jobs only, skip %llu then sample %llu; timings include instrumentation%s",
            (unsigned long long)phase_skip_, (unsigned long long)phase_limit_,
            phase_trace_fp_ ? "; per-job CSV enabled" : "");
    }
    layers_ = layers;
    mb_ = mb;
    res_ = res;
    scratch_.resize(n_threads);
    for (auto& scratch : scratch_) scratch.nibble_rows = 0;
    sync_ = std::make_unique<LayerSync[]>(n_layer);
    iq3_down_cache_.reset();
    if (const char* setting = std::getenv("STRATA_CPU_IQ3_NIBBLE"); setting && *setting && std::strcmp(setting, "0") != 0) {
        SQ_CHECK(std::strcmp(setting, "1") == 0, "STRATA_CPU_IQ3_NIBBLE must be 0 or 1");
        auto candidate = std::make_unique<CpuIq3DownCache>();
        std::string error;
        SQ_CHECK(candidate->build(layers_, n_threads, error), "STRATA_CPU_IQ3_NIBBLE initialization failed: %s", error.c_str());
        iq3_down_cache_ = std::move(candidate);
    }
    quit_ = false;
    const uint32_t gen0 = job_gen_.load();
    try {
        for (int t = 0; t < n_threads; ++t) {
            threads_.emplace_back([this, t, gen0] { worker(t, gen0); });
            if (pin_threads) {
                const int ncpu = (int)std::thread::hardware_concurrency();
                // SMT siblings are adjacent logical processors on Windows (and on most Linux systems with this
                // numbering, cores first is the other common one; pinning is only a hint there): one worker per core.
                const int cpu = ncpu >= 2 ? (2 * t) % ncpu + (2 * t / ncpu) % 2 : 0;
#ifdef _WIN32
                if (cpu < 64) SetThreadAffinityMask(threads_.back().native_handle(), 1ull << cpu);
                SetThreadPriority(threads_.back().native_handle(), THREAD_PRIORITY_ABOVE_NORMAL);
#else
                cpu_set_t set;
                CPU_ZERO(&set);
                CPU_SET(cpu, &set);
                pthread_setaffinity_np(threads_.back().native_handle(), sizeof(set), &set);
#endif
            }
        }
    } catch (const std::exception& exception) {
        stop(); // Join any partially created group before releasing its immutable derived weights.
        die("CPU worker startup failed: %s", exception.what());
    }
}

void CpuMoe::stop() {
    if (phase_trace_fp_) { std::fclose(phase_trace_fp_); phase_trace_fp_ = nullptr; }
    if (phase_workers_fp_) { std::fclose(phase_workers_fp_); phase_workers_fp_ = nullptr; }
    if (threads_.empty()) { iq3_down_cache_.reset(); phase_workers_.reset(); return; }
    quit_ = true;
    job_gen_.fetch_add(1);
    job_gen_.notify_all();
    for (auto& t : threads_) t.join();
    threads_.clear();
    // Normal stop is between drained jobs. Worker join also makes partial-startup cleanup safe; never
    // manufacture a sample from an abandoned job whose end_layer was not called.
    if (phase_enabled_ && !phase_logged_ && phase_profile_.sampled_jobs) phase_log(true);
    phase_workers_.reset();
    iq3_down_cache_.reset();
}

void CpuMoe::dispatch(std::function<void(int)> job) {
    job_ = std::move(job);
    job_left_.store(n_threads_, std::memory_order_relaxed);
    job_gen_.fetch_add(1, std::memory_order_release);
    job_gen_.notify_all();
}

void CpuMoe::wait_job(bool prompt) {
    const int budget = prompt ? prompt_completion_spins_ : completion_spins_;
    if (budget >= 0) {
        int spins = 0;
        for (int left; (left = job_left_.load(std::memory_order_acquire)) > 0;) {
            if (spins < budget) {
                ++spins;
                _mm_pause();
            } else {
                // wait compares again before sleeping: a worker finishing between the load and wait cannot
                // lose the final notification. Intermediate decrements need no wake; only zero completes a
                // job. The sole host scheduler cannot dispatch a new job while it is blocked here (no ABA).
                job_left_.wait(left, std::memory_order_acquire);
            }
        }
        return;
    }
    int spins = 0;
    while (job_left_.load(std::memory_order_acquire) > 0) {
        if (++spins < 1 << 16) _mm_pause();
        else std::this_thread::yield();
    }
}

void CpuMoe::parallel_for_workers(std::function<void(int)> job) {
    SQ_CHECK(n_threads_ > 0 && !threads_.empty() && (bool)job, "CPU worker pool is not ready");
    wait_job();  // Never overwrite a callback while an earlier MoE job still owns it.
    dispatch(std::move(job));
    wait_job();
}

void CpuMoe::worker(int tid, uint32_t seen) {
    for (;;) {
        // Spin briefly (decode steps arrive back to back), then sleep on the generation counter.
        // The cached budget is immutable while workers run. A zero budget waits immediately; wait(seen)
        // checks the generation value again, so dispatch between this load and wait cannot lose a wake-up.
        uint32_t g;
        int spins = 0;
        while ((g = job_gen_.load(std::memory_order_acquire)) == seen) {
            if (spins < worker_spins_) {
                ++spins; // bounded by at most one million; never increases once the worker starts waiting
                _mm_pause();
            } else {
                job_gen_.wait(seen, std::memory_order_acquire);
            }
        }
        seen = g;
        if (quit_) return;
        job_(tid);
        // The release sequence publishes every callback's writes to the host's acquire load/wait. Notify
        // only on the final decrement, after the callback returned, if either immutable wait mode may
        // sleep. stop() joins workers before those settings or the atomic object can be destroyed.
        if (job_left_.fetch_sub(1, std::memory_order_acq_rel) == 1 && notify_completion_)
            job_left_.notify_all();
    }
}

void CpuMoe::reset_sync() {
    for (int l = 0; l < n_layer_; ++l) {
        LayerSync& Y = sync_[l];
        Y.take1.v.store(0, std::memory_order_relaxed);
        Y.done1.v.store(0, std::memory_order_relaxed);
        Y.take2.v.store(0, std::memory_order_relaxed);
        Y.done2.v.store(0, std::memory_order_relaxed);
    }
}

static inline float silu(float x) { return x / (1.0f + std::exp(-x)); }

bool CpuMoe::run_layer(int tid, int layer, const float (*x)[2048], const int* ids, const float (*w)[kMaxT], int n,
                       int n_tok, float (*out)[2048]) {
    return run_layer_impl<false>(tid, layer, x, ids, w, n, n_tok, out);
}

template<bool Profile>
bool CpuMoe::run_layer_impl(int tid, int layer, const float (*x)[2048], const int* ids, const float (*w)[kMaxT], int n,
                            int n_tok, float (*out)[2048]) {
    CpuMoePhaseCounts* phase = nullptr;
    if constexpr (Profile) phase = &phase_workers_[tid].counts;
    struct JobSpan {
        PhaseWorker* w = nullptr;
        uint64_t dispatch = 0, cpu0 = 0;
        JobSpan(PhaseWorker* worker, uint64_t dispatch_ns) : w(worker), dispatch(dispatch_ns) {
            if constexpr (Profile) {
                w->start_offset_ns = phase_now_ns() - dispatch;
                cpu0 = thread_cpu_now_ns();
            }
        }
        ~JobSpan() {
            if constexpr (Profile) {
                w->end_offset_ns = phase_now_ns() - dispatch;
                w->cpu_ns = thread_cpu_now_ns() - cpu0;
            }
        }
    } job_span(Profile ? &phase_workers_[tid] : nullptr, phase_dispatch_ns_);
    PhaseSpan<Profile> worker_time(Profile ? &phase->ns[CpuWorkerTotal] : nullptr);
    // h_ is shared by all layers.  It stays valid while this thread holds an unfinished chunk: the layer cannot
    // complete without it, and the next layer (which overwrites h_) only starts once the GPU has read this layer's
    // result.  So a late thread reads h_ only after it has taken a chunk.
    Scratch& S = scratch_[tid];
    LayerSync& Y = sync_[layer];
    const ExpertLayerDesc& L = layers_[layer];
    const auto* nibble_down = iq3_down_cache_ ? iq3_down_cache_->layer(layer) : nullptr;
    const bool wide_nibble = nibble_down && n_tok > 1 && native_iq::iq3_nibble_wide_enabled();
    // (expert, token) pairs, expert-major; every thread derives the same list
    int pj[kMaxU], pt[kMaxU], first[kMaxU + 1];
    int np = 0;
    for (int j = 0; j < n; ++j) {
        first[j] = np;
        for (int t = 0; t < n_tok; ++t)
            if (w[j][t] != 0.0f && np < kMaxU) { pj[np] = j; pt[np] = t; ++np; }
    }
    first[n] = np;
    const int64_t rb_gu = row_bytes(L.t_gu, kE);
    const DotFn dot_gu = dot_fn(L.t_gu), dot_d = dot_fn(L.t_down);
    const bool reuse_k23 = n_tok > 1 && k23_reuse_enabled();
    const bool k23_gu = reuse_k23 && native_iq::k23_supported(L.t_gu);
    const bool k23_down = reuse_k23 && native_iq::k23_supported(L.t_down);
    const bool reuse_gu = n_tok > 1 && (strata::kernels::cpu::iq256_supported(L.t_gu) || k23_gu);
    const bool reuse_down = n_tok > 1 && (strata::kernels::cpu::iq256_supported(L.t_down) || k23_down);
    const int iq_variant = n_tok > 1 ? strata::kernels::cpu::iq256_variant() : 0;
    const auto chunk_singleton = [iq_variant](uint32_t ty) {
        const bool iq23 = ty == T_IQ2_S || ty == T_IQ3_S;
        return ((iq_variant & strata::kernels::cpu::kIq256InlineRows) && (iq23 || ty == T_IQ4_XS)) ||
               ((iq_variant & strata::kernels::cpu::kIq256CompactScales) && iq23);
    };
    const int R = n * kFF;
    bool have_x = false;
    for (;;) {   // gate/up rows: consecutive rows of one expert per chunk, each row dotted with its tokens
        const int c = Y.take1.v.fetch_add(kChunkGU, std::memory_order_relaxed);
        if (c >= R) break;
        if (!have_x) {
            PhaseSpan<Profile> pack_time(Profile ? &phase->ns[CpuPackGU] : nullptr);
            for (int t = 0; t < n_tok; ++t) quantize_q8k(x[t], S.xq[t], kE);
            if constexpr (Profile) phase->gu_quant_tokens += n_tok;
            have_x = true;
        }
        const int ce = std::min(c + kChunkGU, R);
        for (int g = c; g < ce;) {
            PhaseSpan<Profile> gu_time(Profile ? &phase->ns[CpuGU] : nullptr);
            const int j = g / kFF, r = g % kFF;
            const int nr = std::min(ce - g, kFF - r);
            const uint8_t* blob = L.base + (int64_t)ids[j] * L.blob;
            const int nt = first[j + 1] - first[j];
            if constexpr (Profile) phase->gu_row_pairs += (uint64_t)nr * nt;
            const bool pair_gu_singleton = (iq_variant & strata::kernels::cpu::kIq256PairGu) &&
                (L.t_gu == T_IQ2_S || L.t_gu == T_IQ3_S || L.t_gu == T_IQ4_XS);
            if (reuse_gu && (nt > 1 || (nt == 1 && (chunk_singleton(L.t_gu) || pair_gu_singleton)))) {
                const void* acts[kMaxT];
                float* outputs[kMaxT];
                for (int t = 0; t < nt; ++t) {
                    const int p = first[j] + t;
                    acts[t] = S.xq[pt[p]];
                    outputs[t] = h_[p];
                }
                // Decode each weight block once for all tokens routed to this expert. The Q2/Q3 opt-in
                // retains each token's original scalar accumulation order, including its gate/up activation.
                if (k23_gu)
                    native_iq::k23_gu_rows(L.t_gu, blob, (size_t)rb_gu, (size_t)L.gu_bytes, kE,
                                           acts, nt, outputs, r, r + nr);
                else
                    strata::kernels::cpu::iq256_gu_rows(L.t_gu, blob, (size_t)rb_gu, (size_t)L.gu_bytes, kE,
                                                        acts, nt, outputs, r, r + nr);
            } else {
                for (int row = r; row < r + nr; ++row)
                    for (int p = first[j]; p < first[j + 1]; ++p) {
                        const float gt = dot_gu(blob + row * rb_gu, S.xq[pt[p]], kE / 256);
                        const float up = dot_gu(blob + L.gu_bytes + row * rb_gu, S.xq[pt[p]], kE / 256);
                        h_[p][row] = silu(gt) * up;
                    }
            }
            g += nr;
        }
        Y.done1.v.fetch_add(ce - c, std::memory_order_release);
    }
    const int64_t rb_d = row_bytes(L.t_down, kFF);
    const uint8_t* down[kMaxU];
    for (int j = 0; j < n; ++j) down[j] = L.base + (int64_t)ids[j] * L.blob + 2 * L.gu_bytes;
    bool have_h = false;
    for (;;) {   // down rows: need every gate/up row
        const int c = Y.take2.v.fetch_add(kChunkD, std::memory_order_relaxed);
        if (c >= kE) return false;
        if (!have_h) {
            {
                PhaseSpan<Profile> barrier_time(Profile ? &phase->ns[CpuGUBarrier] : nullptr);
                while (Y.done1.v.load(std::memory_order_acquire) < R) _mm_pause();
            }
            PhaseSpan<Profile> pack_time(Profile ? &phase->ns[CpuPackDown] : nullptr);
            for (int p = 0; p < np; ++p) quantize_q8k(h_[p], S.hq[p], kFF);
            if constexpr (Profile) phase->down_quant_pairs += np;
            have_h = true;
        }
        if (reuse_down || nibble_down) {
            alignas(64) float sums[kMaxT][kChunkD] = {};
            alignas(64) float partial[kMaxT][kChunkD];
            for (int j = 0; j < n; ++j) {
                const int nt = first[j + 1] - first[j];
                {
                    PhaseSpan<Profile> down_time(Profile ? &phase->ns[CpuDown] : nullptr);
                    if constexpr (Profile) phase->down_row_pairs += (uint64_t)nt * kChunkD;
                    if (nibble_down && nt > 0) {
                        const Q8K* acts[kMaxT];
                        float* outputs[kMaxT];
                        for (int t = 0; t < nt; ++t) {
                            acts[t] = S.hq[first[j] + t];
                            outputs[t] = partial[t];
                        }
                        const auto* weights = nibble_down + (size_t)ids[j] * CpuIq3DownCache::BlocksPerExpert +
                                              (size_t)c * (kFF / 256);
                        if (wide_nibble && nt > 1)
                            native_iq::iq3_nibble_rows_wide_prefetch(weights, kFF / 256, acts, nt, outputs, 0, kChunkD);
                        else
                            native_iq::iq3_nibble_rows(weights, kFF / 256, acts, nt, outputs, 0, kChunkD);
                        S.nibble_rows += (uint64_t)nt * kChunkD; // private to this worker; no cost on default path
                    } else if (nt > 1 || (nt == 1 && chunk_singleton(L.t_down))) {
                        const void* acts[kMaxT];
                        float* outputs[kMaxT];
                        for (int t = 0; t < nt; ++t) {
                            acts[t] = S.hq[first[j] + t];
                            outputs[t] = partial[t];
                        }
                        if (k23_down)
                            native_iq::k23_rows(L.t_down, down[j] + c * rb_d, (size_t)rb_d, kFF,
                                                acts, nt, outputs, 0, kChunkD);
                        else
                            strata::kernels::cpu::iq256_rows(L.t_down, down[j] + c * rb_d, (size_t)rb_d, kFF,
                                                             acts, nt, outputs, 0, kChunkD);
                    } else if (nt == 1) {
                        for (int r = 0; r < kChunkD; ++r)
                            partial[0][r] = dot_d(down[j] + (c + r) * rb_d, S.hq[first[j]], kFF / 256);
                    }
                }
                // Preserve the original expert-major accumulation order for each token and row.
                PhaseSpan<Profile> reduce_time(Profile ? &phase->ns[CpuReduce] : nullptr);
                for (int t = 0; t < nt; ++t) {
                    const int token = pt[first[j] + t];
                    for (int r = 0; r < kChunkD; ++r) sums[token][r] += w[j][token] * partial[t][r];
                }
            }
            PhaseSpan<Profile> output_time(Profile ? &phase->ns[CpuReduce] : nullptr);
            for (int t = 0; t < n_tok; ++t)
                std::memcpy(out[t] + c, sums[t], sizeof(float) * kChunkD);
        } else {
            PhaseSpan<Profile> fused_time(Profile ? &phase->ns[CpuDownFused] : nullptr);
            if constexpr (Profile) phase->down_row_pairs += (uint64_t)np * kChunkD;
            for (int r = c; r < c + kChunkD; ++r) {
                float s[kMaxT] = {0.0f, 0.0f, 0.0f, 0.0f};
                for (int p = 0; p < np; ++p) s[pt[p]] += w[pj[p]][pt[p]] * dot_d(down[pj[p]] + r * rb_d, S.hq[p], kFF / 256);
                for (int t = 0; t < n_tok; ++t) out[t][r] = s[t];
            }
        }
        if (Y.done2.v.fetch_add(kChunkD, std::memory_order_acq_rel) + kChunkD == kE) return true;
    }
}

void CpuMoe::begin_step(uint32_t seq, int l0, int l1) {
    timed_out_ = false;
    reset_sync();
    dispatch([this, seq, l0, l1](int tid) {
        for (int l = l0; l < l1; ++l) {
            Mailbox& m = mb_[l];
            int spins = 0;
            auto t0 = std::chrono::steady_clock::now();
            while (m.ready_seq != seq) {
                _mm_pause();
                if (timed_out_.load(std::memory_order_relaxed)) return;
                if ((++spins & 0xFFFF) == 0 &&
                    std::chrono::steady_clock::now() - t0 > std::chrono::seconds(20)) {
                    if (!timed_out_.exchange(true, std::memory_order_relaxed))
                        log("CPU MoE timeout: worker %d layer %d expected seq %u, mailbox ready %u, result done %u, misses %d",
                            tid, l, seq, m.ready_seq, res_[l].done_seq, m.n_miss);
                    return;
                }
            }
            std::atomic_thread_fence(std::memory_order_acquire);
            const int n = m.n_miss;
            if (n <= 0) continue;
            if (run_layer(tid, l, m.x, m.ids, m.w, n, m.n_tok, res_[l].out)) {
                experts_computed += n;
                std::atomic_thread_fence(std::memory_order_release);
                res_[l].done_seq = seq;
            }
        }
    });
}

bool CpuMoe::end_step() {
    wait_job();
    return !timed_out_;
}

void CpuMoe::phase_begin_layer(int layer) {
    SQ_CHECK(!phase_active_, "CPU phase sample must be drained before the next layer");
    const uint64_t index = phase_profile_.seen_jobs++;
    if (index < phase_skip_ || phase_profile_.sampled_jobs >= phase_limit_) return;
    for (int tid = 0; tid < n_threads_; ++tid) {
        phase_workers_[tid].counts = {};
        phase_workers_[tid].start_offset_ns = 0;
        phase_workers_[tid].end_offset_ns = 0;
        phase_workers_[tid].cpu_ns = 0;
    }
    phase_layer_ = layer;
    phase_active_ = true;
    phase_start_ns_ = phase_now_ns();
    phase_dispatch_ns_ = phase_start_ns_;
}

void CpuMoe::phase_end_layer() {
    // Called strictly after wait_job's acquire. Mailbox, worker counters, and output are immutable here;
    // the next dispatch cannot publish new counters while the sole host is merging this sample.
    const uint64_t wall = phase_now_ns() - phase_start_ns_;
    const uint64_t job_index = phase_profile_.sampled_jobs;
    const Mailbox& m = mb_[phase_layer_];
    const ExpertLayerDesc& layer = layers_[phase_layer_];
    auto& groups = phase_profile_.groups;
    size_t index = 0;
    for (; index < groups.size(); ++index)
        if (groups[index].kind == job_kind_ && groups[index].n_tok == m.n_tok &&
            groups[index].t_gu == layer.t_gu && groups[index].t_down == layer.t_down) break;
    if (index == groups.size()) {
        CpuMoePhaseGroup group;
        group.kind = job_kind_;
        group.n_tok = m.n_tok; group.t_gu = layer.t_gu; group.t_down = layer.t_down;
        groups.push_back(group);
    }
    auto& group = groups[index];
    ++group.jobs;
    group.dispatch_to_join_ns += wall;
    group.missing_experts += m.n_miss;
    uint64_t job_pairs = 0;
    for (int j = 0; j < m.n_miss; ++j) {
        int nt = 0;
        for (int t = 0; t < m.n_tok; ++t) nt += m.w[j][t] != 0.f;
        ++group.expert_nt[nt];
        group.routed_pairs += nt;
        job_pairs += nt;
    }
    uint64_t max_worker = 0;
    uint64_t first_start = UINT64_MAX, last_start = 0, end_min = UINT64_MAX, last_end = 0, cpu_sum = 0;
    uint64_t cpu_max = 0, cpu_min = UINT64_MAX;
    int entered = 0;
    for (int tid = 0; tid < n_threads_; ++tid) {
        const auto& source = phase_workers_[tid].counts;
        for (int p = 0; p < CpuPhaseCount; ++p) group.workers.ns[p] += source.ns[p];
        group.workers.gu_quant_tokens += source.gu_quant_tokens;
        group.workers.down_quant_pairs += source.down_quant_pairs;
        group.workers.gu_row_pairs += source.gu_row_pairs;
        group.workers.down_row_pairs += source.down_row_pairs;
        max_worker = std::max(max_worker, source.ns[CpuWorkerTotal]);
        const PhaseWorker& worker = phase_workers_[tid];
        if (worker.start_offset_ns == 0) continue;   // worker never entered run_layer for this job
        first_start = std::min(first_start, worker.start_offset_ns);
        last_start = std::max(last_start, worker.start_offset_ns);
        end_min = std::min(end_min, worker.end_offset_ns);
        last_end = std::max(last_end, worker.end_offset_ns);
        cpu_sum += worker.cpu_ns;
        cpu_max = std::max(cpu_max, worker.cpu_ns);
        cpu_min = std::min(cpu_min, worker.cpu_ns);
        ++entered;
        if (phase_workers_fp_)
            std::fprintf(phase_workers_fp_, "%llu,%d,%d,%d,%d,%llu,%llu,%llu\n",
                         (unsigned long long)job_index, job_kind_, phase_layer_, m.n_tok, tid,
                         (unsigned long long)worker.start_offset_ns,
                         (unsigned long long)worker.end_offset_ns, (unsigned long long)worker.cpu_ns);
    }
    if (phase_workers_fp_) std::fflush(phase_workers_fp_);
    if (first_start == UINT64_MAX) first_start = 0;
    if (end_min == UINT64_MAX) end_min = 0;
    if (cpu_min == UINT64_MAX) cpu_min = 0;
    group.max_worker_ns += max_worker;
    group.dispatch_to_first_start_ns += first_start;
    group.start_spread_ns += last_start - first_start;
    group.end_spread_ns += last_end - end_min;
    group.dispatch_to_last_end_ns += last_end;
    group.worker_cpu_ns += cpu_sum;
    group.worker_jobs += (uint64_t)entered;
    group.max_worker_cpu_ns += cpu_max;
    group.min_worker_cpu_ns += cpu_min;
    if (phase_trace_fp_) {
        std::fprintf(phase_trace_fp_,
                     "%llu,%llu,%d,%d,%d,%d,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu\n",
                     (unsigned long long)job_index, (unsigned long long)phase_dispatch_ns_, job_kind_,
                     phase_layer_, m.n_tok, m.n_miss, (unsigned long long)job_pairs,
                     (unsigned long long)first_start, (unsigned long long)(last_start - first_start),
                     (unsigned long long)end_min, (unsigned long long)last_end, (unsigned long long)wall,
                     (unsigned long long)cpu_sum, (unsigned long long)cpu_max, (unsigned long long)cpu_min,
                     (unsigned long long)max_worker);
        std::fflush(phase_trace_fp_);
    }
    phase_active_ = false;
    ++phase_profile_.sampled_jobs;
    if (phase_profile_.sampled_jobs == phase_limit_) {
        phase_profile_.complete = true;
        phase_log(false);
    }
}

void CpuMoe::phase_log(bool partial) {
    phase_logged_ = true;
    log("CPU_PHASE_SUMMARY jobs=%llu seen=%llu skip=%llu partial=%d threads=%d; phase times are summed worker elapsed durations including waiting/preemption, wall is dispatch-to-join; event-layer jobs only",
        (unsigned long long)phase_profile_.sampled_jobs, (unsigned long long)phase_profile_.seen_jobs,
        (unsigned long long)phase_skip_, partial ? 1 : 0, n_threads_);
    for (const auto& group : phase_profile_.groups) {
        const auto& ns = group.workers.ns;
        const double jobs = (double)std::max<uint64_t>(group.jobs, 1);
        const double cpu_mean_us = group.worker_jobs ? (double)group.worker_cpu_ns / group.worker_jobs / 1000.0 : 0.0;
        const double cpu_max_us = (double)group.max_worker_cpu_ns / jobs / 1000.0;
        const double cpu_min_us = (double)group.min_worker_cpu_ns / jobs / 1000.0;
        const double util = group.dispatch_to_join_ns
            ? (double)group.worker_cpu_ns / ((double)group.dispatch_to_join_ns * n_threads_) : 0.0;
        log("CPU_PHASE kind=%d T=%d GU=%u DOWN=%u jobs=%llu missing=%llu pairs=%llu expertNT0..4=%llu,%llu,%llu,%llu,%llu "
            "wall_ms=%.6f max_worker_ms=%.6f worker_jobs=%llu cpu_mean_us=%.1f cpu_max_mean_us=%.1f "
            "cpu_min_mean_us=%.1f max_over_mean=%.3f worker_util=%.3f start_us=%.3f start_spread_us=%.1f "
            "end_spread_us=%.1f last_end_us=%.3f pack_gu_ms=%.6f gu_ms=%.6f gu_barrier_ms=%.6f "
            "pack_down_ms=%.6f down_ms=%.6f reduce_ms=%.6f down_fused_ms=%.6f "
            "gu_quant_tokens=%llu down_quant_pairs=%llu gu_row_pairs=%llu down_row_pairs=%llu",
            group.kind, group.n_tok, group.t_gu, group.t_down, (unsigned long long)group.jobs,
            (unsigned long long)group.missing_experts, (unsigned long long)group.routed_pairs,
            (unsigned long long)group.expert_nt[0], (unsigned long long)group.expert_nt[1],
            (unsigned long long)group.expert_nt[2], (unsigned long long)group.expert_nt[3],
            (unsigned long long)group.expert_nt[4],
            group.dispatch_to_join_ns * 1e-6, group.max_worker_ns * 1e-6,
            (unsigned long long)group.worker_jobs, cpu_mean_us, cpu_max_us, cpu_min_us,
            cpu_mean_us > 0 ? cpu_max_us / cpu_mean_us : 0.0, util,
            group.dispatch_to_first_start_ns / jobs / 1000.0, group.start_spread_ns / jobs / 1000.0,
            group.end_spread_ns / jobs / 1000.0, group.dispatch_to_last_end_ns / jobs / 1000.0,
            ns[CpuPackGU] * 1e-6, ns[CpuGU] * 1e-6, ns[CpuGUBarrier] * 1e-6,
            ns[CpuPackDown] * 1e-6, ns[CpuDown] * 1e-6, ns[CpuReduce] * 1e-6, ns[CpuDownFused] * 1e-6,
            (unsigned long long)group.workers.gu_quant_tokens, (unsigned long long)group.workers.down_quant_pairs,
            (unsigned long long)group.workers.gu_row_pairs, (unsigned long long)group.workers.down_row_pairs);
    }
}

void CpuMoe::begin_layer(int layer) {
    SQ_CHECK(layer >= 0 && layer < n_layer_, "invalid CPU MoE layer %d", layer);
    const Mailbox& m = mb_[layer];
    SQ_CHECK(m.n_tok >= 1 && m.n_tok <= kMaxT && m.n_miss >= 0 && m.n_miss <= kMaxU,
             "invalid CPU MoE mailbox: layer %d, tokens %d, misses %d", layer, m.n_tok, m.n_miss);
    // The caller finished the previous job before dispatching this one.  This acquire/release worker handoff,
    // not a GPU-visible spin flag, establishes when the host may upload the result.
    reset_sync();
    if (phase_enabled_) phase_begin_layer(layer);
    if (phase_active_) {
        dispatch([this, layer](int tid) {
            const Mailbox& m = mb_[layer];
            if (m.n_miss > 0 && run_layer_impl<true>(tid, layer, m.x, m.ids, m.w, m.n_miss, m.n_tok, res_[layer].out))
                experts_computed += m.n_miss;
        });
        return;
    }
    dispatch([this, layer](int tid) {
        const Mailbox& m = mb_[layer];
        if (m.n_miss > 0 && run_layer(tid, layer, m.x, m.ids, m.w, m.n_miss, m.n_tok, res_[layer].out))
            experts_computed += m.n_miss;
    });
}

void CpuMoe::end_layer(bool prompt) {
    wait_job(prompt);
    if (phase_active_) phase_end_layer();
}

void CpuMoe::compute_layer_sync(int layer, const float* x, const int* ids, const float* w, int n, float* out) {
    auto xx = std::make_unique<float[][2048]>(1);
    auto oo = std::make_unique<float[][2048]>(1);
    std::memcpy(xx[0], x, sizeof(float) * 2048);
    float ww[kMaxU][kMaxT] = {};
    for (int j = 0; j < n; ++j) ww[j][0] = w[j];
    reset_sync();
    dispatch([&, this](int tid) { run_layer(tid, layer, xx.get(), ids, ww, n, 1, oo.get()); });
    wait_job();
    std::memcpy(out, oo[0], sizeof(float) * 2048);
}

}  // namespace sq
