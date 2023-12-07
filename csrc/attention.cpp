#include <torch/extension.h>
#include <c10/util/Optional.h>

void paged_attention_v1(
  torch::Tensor& out,
  torch::Tensor& query,
  torch::Tensor& key_cache,
  torch::Tensor& value_cache,
  float scale,
  torch::Tensor& context_lens,
  int block_size,
  int max_context_len,
  const c10::optional<torch::Tensor>& alibi_slopes,
  torch::Tensor& key,           // [num_tokens, num_heads, head_size]
  torch::Tensor& value,         // [num_tokens, num_heads, head_size]
  torch::Tensor& position,      // [num_tokens]
  // const int num_heads,
  // const int head_size,
  // const int block_size,
  const int x,
  const int max_num_blocks_per_seq
  );


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
    "paged_attention_v1",
    &paged_attention_v1,
    "Compute the attention between an input query and the cached keys/values using PagedAttention.");
}
