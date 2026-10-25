#include "dspark_cpu.hpp"
#include "quant.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using namespace sq;

namespace {

struct Mat {
    int rows = 0, cols = 0;
    std::vector<float> w;
};

Mat load_mat(const GgufTensor& t) {
    Mat m;
    m.cols = (int)t.ne[0];
    m.rows = (int)(t.n_elements() / m.cols);
    m.w.resize((size_t)t.n_elements());
    dequant_row(t.type, t.data, m.w.data(), t.n_elements());
    return m;
}

void matvec(const Mat& m, const float* x, float* y) {
    for (int r = 0; r < m.rows; ++r) {
        const float* w = m.w.data() + (size_t)r * m.cols;
        double s = 0;
        for (int c = 0; c < m.cols; ++c) s += (double)w[c] * x[c];
        y[r] = (float)s;
    }
}

void rmsnorm(float* x, const std::vector<float>& w, int n, float eps = 1e-6f) {
    double s = 0;
    for (int i = 0; i < n; ++i) s += (double)x[i] * x[i];
    const float d = 1.0f / std::sqrt((float)(s / n) + eps);
    for (int i = 0; i < n; ++i) x[i] = x[i] * d * w[i];
}

void rope_neox(float* x, int pos, int n_head, int hd) {
    // Full head_dim NeoX, denominator 256; the draft uses 256-dim heads.
    for (int h = 0; h < n_head; ++h) {
        float* v = x + (size_t)h * hd;
        for (int i = 0; i < hd / 2; ++i) {
            const double ang = pos * std::pow(1e7, -2.0 * i / hd);
            const float c = (float)std::cos(ang), s = (float)std::sin(ang);
            const float a = v[i], b = v[i + hd / 2];
            v[i] = a * c - b * s;
            v[i + hd / 2] = a * s + b * c;
        }
    }
}

float dot(const float* a, const float* b, int n) {
    double s = 0;
    for (int i = 0; i < n; ++i) s += (double)a[i] * b[i];
    return (float)s;
}

struct Layer {
    Mat q, k, v, o, gate, up, down;
    std::vector<float> an, qn, kn, fn;
};

struct Ref {
    int n_layer = 5, ne = 2048, nf = 6144, nh = 8, nkv = 2, hd = 256, mask = 248076;
    std::vector<Layer> L;
    Mat fc;
    Mat w1, w2;
    std::vector<float> encn, outn, cw;
    float cb = 0;
    std::vector<int> taps_layers;

    bool load(const GgufFile& g, std::string& err) {
        try {
            n_layer = (int)g.i64("dflash.block_count");
            ne = (int)g.i64("dflash.embedding_length");
            nf = (int)g.i64("dflash.feed_forward_length");
            nh = (int)g.i64("dflash.attention.head_count");
            nkv = (int)g.i64("dflash.attention.head_count_kv");
            hd = (int)g.i64("dflash.attention.key_length");
            mask = (int)g.i64("tokenizer.ggml.mask_token_id");
            for (double v : g.arr("dflash.target_layers")) taps_layers.push_back((int)v);
            L.resize(n_layer);
            for (int i = 0; i < n_layer; ++i) {
                const std::string p = "blk." + std::to_string(i) + ".";
                Layer& l = L[i];
                l.q = load_mat(g.need(p + "attn_q.weight"));
                l.k = load_mat(g.need(p + "attn_k.weight"));
                l.v = load_mat(g.need(p + "attn_v.weight"));
                l.o = load_mat(g.need(p + "attn_output.weight"));
                l.gate = load_mat(g.need(p + "ffn_gate.weight"));
                l.up = load_mat(g.need(p + "ffn_up.weight"));
                l.down = load_mat(g.need(p + "ffn_down.weight"));
                auto vec = [&](const std::string& n) {
                    const GgufTensor& t = g.need(n);
                    std::vector<float> v((size_t)t.n_elements());
                    dequant_row(t.type, t.data, v.data(), t.n_elements());
                    return v;
                };
                l.an = vec(p + "attn_norm.weight");
                l.qn = vec(p + "attn_q_norm.weight");
                l.kn = vec(p + "attn_k_norm.weight");
                l.fn = vec(p + "ffn_norm.weight");
            }
            fc = load_mat(g.need("fc.weight"));
            w1 = load_mat(g.need("markov_w1.weight"));
            w2 = load_mat(g.need("markov_w2.weight"));
            auto vec = [&](const std::string& n) {
                const GgufTensor& t = g.need(n);
                std::vector<float> v((size_t)t.n_elements());
                dequant_row(t.type, t.data, v.data(), t.n_elements());
                return v;
            };
            encn = vec("enc.output_norm.weight");
            outn = vec("output_norm.weight");
            cw = vec("conf_proj.weight");
            std::vector<float> cbb = vec("conf_proj.bias");
            cb = cbb.empty() ? 0 : cbb[0];
            return true;
        } catch (const std::exception& e) {
            err = e.what();
            return false;
        }
    }
};

}  // namespace

bool sq::dspark_cpu_draft(const GgufFile& dg, const GgufFile& tg,
                          const std::vector<float>& taps_rows, int prompt_len,
                          int anchor, int pos0, std::vector<int>& out, std::string& err) {
    Ref ref;
    if (!ref.load(dg, err)) return false;
    const int E = ref.ne, P = prompt_len;
    const int T = 4;
    std::vector<std::vector<float>> taps((size_t)P, std::vector<float>((size_t)ref.taps_layers.size() * E));
    for (int p = 0; p < P; ++p)
        std::memcpy(taps[p].data(), taps_rows.data() + (size_t)p * taps[p].size(), taps[p].size() * 4);
    // Encoder: g = fc(taps), enc norm; per-layer K/V at prompt positions.
    std::vector<std::vector<float>> g(P, std::vector<float>(E));
    for (int p = 0; p < P; ++p) {
        matvec(ref.fc, taps[p].data(), g[p].data());
        rmsnorm(g[p].data(), ref.encn, E);
    }
    std::vector<std::vector<float>> ek(ref.n_layer), ev(ref.n_layer);
    for (int l = 0; l < ref.n_layer; ++l) {
        ek[l].resize((size_t)P * ref.nkv * ref.hd);
        ev[l].resize((size_t)P * ref.nkv * ref.hd);
        for (int p = 0; p < P; ++p) {
            std::vector<float> k((size_t)ref.nkv * ref.hd), v((size_t)ref.nkv * ref.hd);
            matvec(ref.L[l].k, g[p].data(), k.data());
            matvec(ref.L[l].v, g[p].data(), v.data());
            for (int h = 0; h < ref.nkv; ++h) {
                float* kh = k.data() + h * ref.hd;
                rmsnorm(kh, ref.L[l].kn, ref.hd);
                rope_neox(kh, p, 1, ref.hd);
            }
            std::memcpy(&ek[l][(size_t)p * ref.nkv * ref.hd], k.data(), k.size() * 4);
            std::memcpy(&ev[l][(size_t)p * ref.nkv * ref.hd], v.data(), v.size() * 4);
        }
    }

    const GgufTensor& emb = tg.need("token_embd.weight");
    const int64_t emb_row = emb.bytes / emb.ne[1];
    const GgufTensor& outw = tg.need("output.weight");
    const int64_t out_row = outw.bytes / outw.ne[1];
    std::vector<std::vector<float>> h((size_t)T, std::vector<float>(E));
    int ids[4] = {anchor, ref.mask, ref.mask, ref.mask};
    for (int t = 0; t < T; ++t) {
        dequant_row(emb.type, emb.data + (int64_t)ids[t] * emb_row, h[t].data(), E);
    }
    std::vector<float> pos_emb(E);
    for (int l = 0; l < ref.n_layer; ++l) {
        Layer& L = ref.L[l];
        std::vector<std::vector<float>> q((size_t)T, std::vector<float>(E)), k((size_t)T, std::vector<float>(ref.nkv * ref.hd)),
            v((size_t)T, std::vector<float>(ref.nkv * ref.hd)), hn((size_t)T, std::vector<float>(E));
        for (int t = 0; t < T; ++t) {
            hn[t] = h[t];
            rmsnorm(hn[t].data(), L.an, E);
            matvec(L.q, hn[t].data(), q[t].data());
            matvec(L.k, hn[t].data(), k[t].data());
            matvec(L.v, hn[t].data(), v[t].data());
            for (int hh = 0; hh < ref.nh; ++hh) {
                float* qh = q[t].data() + hh * ref.hd;
                rmsnorm(qh, L.qn, ref.hd);
                rope_neox(qh, pos0 + t, 1, ref.hd);
            }
            for (int hh = 0; hh < ref.nkv; ++hh) {
                float* kh = k[t].data() + hh * ref.hd;
                rmsnorm(kh, L.kn, ref.hd);
                rope_neox(kh, pos0 + t, 1, ref.hd);
            }
        }
        // Full non-causal attention over encoder + block.
        std::vector<std::vector<float>> att((size_t)T, std::vector<float>(E));
        const int NK = P + T;
        for (int t = 0; t < T; ++t) {
            for (int hh = 0; hh < ref.nh; ++hh) {
                const int kv = hh / (ref.nh / ref.nkv);
                std::vector<float> sc(NK);
                for (int j = 0; j < NK; ++j) {
                    const float* kk = j < P ? &ek[l][(size_t)j * ref.nkv * ref.hd + kv * ref.hd]
                                            : &k[j - P][kv * ref.hd];
                    sc[j] = dot(&q[t][hh * ref.hd], kk, ref.hd) / 16.0f;
                }
                float mx = *std::max_element(sc.begin(), sc.end());
                double sum = 0;
                for (float& s : sc) { s = std::exp(s - mx); sum += s; }
                float* av = &att[t][hh * ref.hd];
                std::fill(av, av + ref.hd, 0.0f);
                for (int j = 0; j < NK; ++j) {
                    const float* vv = j < P ? &ev[l][(size_t)j * ref.nkv * ref.hd + kv * ref.hd]
                                            : &v[j - P][kv * ref.hd];
                    const float p = (float)(sc[j] / sum);
                    for (int d = 0; d < ref.hd; ++d) av[d] += p * vv[d];
                }
            }
        }
        for (int t = 0; t < T; ++t) {
            std::vector<float> o(E);
            matvec(L.o, att[t].data(), o.data());
            for (int i = 0; i < E; ++i) h[t][i] += o[i];
            std::vector<float> ff(E);
            hn[t] = h[t];
            rmsnorm(hn[t].data(), L.fn, E);
            std::vector<float> gate((size_t)ref.nf), up((size_t)ref.nf), act((size_t)ref.nf);
            matvec(L.gate, hn[t].data(), gate.data());
            matvec(L.up, hn[t].data(), up.data());
            for (int i = 0; i < ref.nf; ++i) act[i] = gate[i] / (1.0f + std::exp(-gate[i])) * up[i];
            matvec(L.down, act.data(), ff.data());
            for (int i = 0; i < E; ++i) h[t][i] += ff[i];
        }
    }
    // output_norm and target lm_head logits.
    std::vector<std::vector<float>> x((size_t)T, std::vector<float>(E));
    for (int t = 0; t < T; ++t) {
        x[t] = h[t];
        rmsnorm(x[t].data(), ref.outn, E);
    }
    const int V = (int)outw.ne[1];
    std::vector<std::vector<float>> logits((size_t)T, std::vector<float>(V));
    std::vector<float> row(E);
    for (int r = 0; r < V; ++r) {
        dequant_row(outw.type, outw.data + (int64_t)r * out_row, row.data(), E);
        for (int t = 0; t < T; ++t) logits[t][r] = dot(row.data(), x[t].data(), E);
    }
    // Markov chain + confidence.
    int prev = anchor;
    for (int t = 0; t < T; ++t) {
        std::vector<float> r(256);
        std::memcpy(r.data(), ref.w1.w.data() + (size_t)prev * 256, 256 * sizeof(float));
        std::vector<float> bias(V);
        float* lp = logits[t].data();
        for (int v = 0; v < V; ++v) {
            const float* wr = ref.w2.w.data() + (size_t)v * 256;
            double s = 0;
            for (int i = 0; i < 256; ++i) s += (double)wr[i] * r[i];
            bias[v] = lp[v] + (float)s;
        }
        int best = (int)(std::max_element(bias.begin(), bias.end()) - bias.begin());
        std::vector<float> feat(E + 256);
        std::memcpy(feat.data(), x[t].data(), E * 4);
        std::memcpy(feat.data() + E, r.data(), 256 * 4);
        float conf = 1.0f / (1.0f + std::exp(-(dot(feat.data(), ref.cw.data(), E + 256) + ref.cb)));
        std::printf("draft %d: %d conf %.3f\n", t, best, conf);
        out.push_back(best);
        prev = best;
    }

    return true;
}

bool sq::dspark_markov_host(const GgufFile& g, const std::vector<float>& xnorm,
                            const std::vector<float>& logits, int anchor, int T, int n_vocab,
                            std::vector<int>& out, std::vector<float>& conf, std::string& err) {
    try {
        const GgufTensor& w1 = g.need("markov_w1.weight");
        const GgufTensor& w2 = g.need("markov_w2.weight");
        const GgufTensor& cwt = g.need("conf_proj.weight");
        const GgufTensor& cbt = g.need("conf_proj.bias");
        if ((int)w1.ne[1] != n_vocab || (int)w2.ne[1] != n_vocab)
            throw std::runtime_error("DSpark markov vocab mismatch");
        const int64_t r1 = w1.bytes / w1.ne[1], r2 = w2.bytes / w2.ne[1];
        std::vector<float> cw((size_t)cwt.n_elements()), cb((size_t)cbt.n_elements());
        dequant_row(cwt.type, cwt.data, cw.data(), cwt.n_elements());
        dequant_row(cbt.type, cbt.data, cb.data(), cbt.n_elements());
        const float bias0 = cb.empty() ? 0.0f : cb[0];
        std::vector<float> r(256), wrow(256);
        out.clear(); conf.clear();
        int prev = anchor;
        for (int t = 0; t < T; ++t) {
            dequant_row(w1.type, w1.data + (int64_t)prev * r1, r.data(), 256);
            int best = 0;
            float bestv = -INFINITY;
            for (int v = 0; v < n_vocab; ++v) {
                dequant_row(w2.type, w2.data + (int64_t)v * r2, wrow.data(), 256);
                double dot = 0;
                for (int i = 0; i < 256; ++i) dot += (double)wrow[i] * r[i];
                const float val = logits[(size_t)t * n_vocab + v] + (float)dot;
                if (val > bestv) { bestv = val; best = v; }
            }
            double c = bias0;
            for (int i = 0; i < 2048; ++i) c += (double)xnorm[(size_t)t * 2048 + i] * cw[i];
            for (int i = 0; i < 256; ++i) c += (double)r[i] * cw[2048 + i];
            conf.push_back(1.0f / (1.0f + std::exp(-(float)c)));
            out.push_back(best);
            prev = best;
        }
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}
