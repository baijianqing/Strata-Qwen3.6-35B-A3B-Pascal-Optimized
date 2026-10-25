#pragma once
#include "gguf.hpp"
#include <string>
#include <vector>

namespace sq {

/// Run the MiniCPM DSpark 4-draft CPU forward.
/// taps_rows: [n_rows][5*n_embd] where the five tap vectors are in the drafter's target-layer order.
/// anchor: the last committed token; pos0: the first draft position (usually n_rows).
bool dspark_cpu_draft(const GgufFile& drafter, const GgufFile& target,
                      const std::vector<float>& taps_rows, int n_rows,
                      int anchor, int pos0, std::vector<int>& out, std::string& err);

/// Markov/confidence chain on host logits from the GPU forward.  xnorm is [T*n_embd], logits [T*n_vocab].
bool dspark_markov_host(const GgufFile& drafter, const std::vector<float>& xnorm,
                        const std::vector<float>& logits, int anchor, int T, int n_vocab,
                        std::vector<int>& out, std::vector<float>& conf, std::string& err);

}  // namespace sq
