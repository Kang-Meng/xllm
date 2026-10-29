/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <cstdint>

extern "C" void kpool_pack_integer_metadata(void* queue,
                                            const void* table,
                                            void* packed_table,
                                            int64_t table_count,
                                            const void* positions,
                                            void* packed_positions,
                                            int64_t position_count,
                                            const void* rows,
                                            void* packed_rows,
                                            int64_t row_count,
                                            const void* starts,
                                            void* packed_starts,
                                            int64_t start_count,
                                            int32_t position_max,
                                            int32_t clamp_positive_positions);

extern "C" void kpool_update_wide_positions(void* queue,
                                            const void* key,
                                            const void* gate,
                                            const void* ape,
                                            const void* matrix,
                                            void* cache,
                                            void* tail,
                                            const void* ids,
                                            const void* read_ids,
                                            const void* table,
                                            const void* positions,
                                            const void* starts,
                                            const void* rows,
                                            void* workspace,
                                            int32_t batch,
                                            int32_t tokens,
                                            int32_t table_width,
                                            int32_t ring_size,
                                            int32_t inherit,
                                            int32_t prefill);

extern "C" void kpool_update_wide_positions_i32(void* queue,
                                                const void* key,
                                                const void* gate,
                                                const void* ape,
                                                const void* matrix,
                                                void* cache,
                                                void* tail,
                                                const void* ids,
                                                const void* read_ids,
                                                const void* table,
                                                const void* positions,
                                                const void* starts,
                                                const void* rows,
                                                void* workspace,
                                                int32_t batch,
                                                int32_t tokens,
                                                int32_t table_width,
                                                int32_t ring_size,
                                                int32_t inherit,
                                                int32_t prefill);

extern "C" void kpool_update_tokens_workspace(void* queue,
                                              const void* key,
                                              const void* gate,
                                              const void* ape,
                                              const void* matrix,
                                              void* cache,
                                              void* tail,
                                              const void* ids,
                                              const void* read_ids,
                                              const void* table,
                                              const void* positions,
                                              const void* starts,
                                              void* workspace,
                                              int32_t batch,
                                              int32_t table_width,
                                              int32_t ring_size,
                                              int32_t inherit,
                                              int32_t tokens);

extern "C" void kpool_update_tokens_workspace_i32(void* queue,
                                                  const void* key,
                                                  const void* gate,
                                                  const void* ape,
                                                  const void* matrix,
                                                  void* cache,
                                                  void* tail,
                                                  const void* ids,
                                                  const void* read_ids,
                                                  const void* table,
                                                  const void* positions,
                                                  const void* starts,
                                                  void* workspace,
                                                  int32_t batch,
                                                  int32_t table_width,
                                                  int32_t ring_size,
                                                  int32_t inherit,
                                                  int32_t tokens);

extern "C" void kpool_prefill_workspace(void* queue,
                                        const void* key,
                                        const void* gate,
                                        const void* ape,
                                        const void* matrix,
                                        void* cache,
                                        void* tail,
                                        const void* ids,
                                        const void* read_ids,
                                        const void* table,
                                        const void* positions,
                                        const void* starts,
                                        const void* rows,
                                        void* workspace,
                                        int32_t batch,
                                        int32_t tokens,
                                        int32_t table_width,
                                        int32_t ring_size,
                                        int32_t inherit);

extern "C" void kpool_prefill_workspace_i32(void* queue,
                                            const void* key,
                                            const void* gate,
                                            const void* ape,
                                            const void* matrix,
                                            void* cache,
                                            void* tail,
                                            const void* ids,
                                            const void* read_ids,
                                            const void* table,
                                            const void* positions,
                                            const void* starts,
                                            const void* rows,
                                            void* workspace,
                                            int32_t batch,
                                            int32_t tokens,
                                            int32_t table_width,
                                            int32_t ring_size,
                                            int32_t inherit);

extern "C" void kpool_score_bounded(void* queue,
                                    const void* query,
                                    const void* weights,
                                    const void* cache,
                                    const void* table,
                                    const void* positions,
                                    const void* rows,
                                    void* output,
                                    int32_t tokens,
                                    int32_t capacity,
                                    int32_t table_width,
                                    int32_t cache_blocks,
                                    int32_t batch,
                                    float scale);

extern "C" void kpool_score_unshared_bounded(void* queue,
                                             const void* query,
                                             const void* weights,
                                             const void* cache,
                                             const void* table,
                                             const void* positions,
                                             const void* rows,
                                             void* output,
                                             int32_t tokens,
                                             int32_t capacity,
                                             int32_t table_width,
                                             int32_t cache_blocks,
                                             int32_t batch,
                                             float scale);

extern "C" void kpool_expand_bounded(void* queue,
                                     const void* selected,
                                     const void* positions,
                                     const void* rows,
                                     const void* table,
                                     void* output,
                                     void* lengths,
                                     int32_t tokens,
                                     int32_t table_width,
                                     int32_t batch,
                                     int32_t append_tail);

extern "C" void kpool_prefill_expand_bounded(void* queue,
                                             const void* selected,
                                             const void* positions,
                                             const void* rows,
                                             const void* table,
                                             void* output,
                                             void* lengths,
                                             int32_t tokens,
                                             int32_t table_width,
                                             int32_t batch,
                                             int32_t append_tail);

extern "C" void kpool_rows(void* queue,
                           const void* starts,
                           void* output,
                           int32_t batch,
                           int32_t tokens,
                           int32_t bits);

extern "C" void kpool_stash_workspace(void* queue,
                                      const void* key,
                                      const void* gate,
                                      void* tail,
                                      const void* ids,
                                      const void* read_ids,
                                      const void* positions,
                                      const void* starts,
                                      void* workspace,
                                      int32_t batch,
                                      int32_t ring_size,
                                      int32_t bits);

extern "C" void kpool_generic_tail_workspace_bounded(void* queue,
                                                     const void* key,
                                                     const void* gate,
                                                     void* tail,
                                                     const void* ids,
                                                     const void* read_ids,
                                                     const void* positions,
                                                     const void* starts,
                                                     void* workspace,
                                                     int32_t batch,
                                                     int32_t ring_size,
                                                     int32_t dimension,
                                                     int32_t flags,
                                                     int64_t tail_states);

extern "C" bool kpool_score_pipelined(void* queue,
                                      const void* query,
                                      const void* weights,
                                      const void* cache,
                                      const void* table,
                                      const void* positions,
                                      const void* rows,
                                      void* output,
                                      int32_t tokens,
                                      int32_t capacity,
                                      int32_t table_width,
                                      int32_t cache_blocks,
                                      int32_t batch,
                                      float scale,
                                      int32_t window);

extern "C" bool kpool_expand_decode(void* queue,
                                    const void* selected,
                                    const void* positions,
                                    const void* rows,
                                    const void* table,
                                    void* output,
                                    void* lengths,
                                    int32_t tokens,
                                    int32_t table_width,
                                    int32_t batch,
                                    int32_t append_tail);
extern "C" bool kpool_cache_positions(void* queue,
                                      const void* rows,
                                      const void* tail_ids,
                                      int32_t tail_bits,
                                      int32_t states,
                                      const void* positions,
                                      const void* slot_mapping,
                                      int32_t slot_bits,
                                      void* output,
                                      int32_t tokens);
extern "C" bool kpool_score_prefill(void* queue,
                                    const void* query,
                                    const void* weights,
                                    const void* cache,
                                    const void* table,
                                    const void* positions,
                                    const void* rows,
                                    void* output,
                                    int32_t tokens,
                                    int32_t capacity,
                                    int32_t table_width,
                                    int32_t cache_blocks,
                                    int32_t batch,
                                    float scale);

// Top-k selection over score rows (kpool_select.mlu): per row the IDs
// torch.topk(count) selects, read from the live prefix
// [0, (positions[row] + 1) / 4) (every pool when positions is null).
// False: unsupported shape.
extern "C" bool kpool_select_topk_short(void* queue,
                                        const void* scores,
                                        const void* positions,
                                        void* output,
                                        int32_t rows,
                                        int32_t pools,
                                        int32_t stride,
                                        int32_t count);

// Rows with more than long_pools live pools may be selected by a cluster.
extern "C" bool kpool_select_topk_hybrid(void* queue,
                                         const void* scores,
                                         const void* positions,
                                         void* output,
                                         int32_t rows,
                                         int32_t pools,
                                         int32_t stride,
                                         int32_t count,
                                         int32_t long_pools);

// Parts per row and their segment length for kpool_select_topk_parts (0:
// unsupported). Workspace with more than one part: 2 * rows * parts * count
// floats and 2 * rows * parts int32.
extern "C" int32_t kpool_select_parts(int32_t rows,
                                      int32_t pools,
                                      int32_t count,
                                      int32_t* segment);

extern "C" bool kpool_select_topk_parts(void* queue,
                                        const void* scores,
                                        const void* positions,
                                        void* output,
                                        void* workspace,
                                        int32_t rows,
                                        int32_t pools,
                                        int32_t stride,
                                        int32_t count,
                                        int32_t parts,
                                        int32_t segment);
