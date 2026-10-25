# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU metadata contracts for a hypothetical mounted V4.1 prefix.

Does not enable prefix caching or exercise C++ allocation or Huawei kernels.
These tests constrain the metadata that a future C1/C2 restore must supply.
"""

from __future__ import annotations

import unittest

import torch

from xllm.python.attention.backend import LayerCache
from xllm.python.attention.csa2_attention import Csa2AttentionBackend
from xllm.python.attention.csa_attention import _scatter_by_slot
from xllm.python.attention.dsa_metadata import (
    DSA_CACHE_SLIDING_WINDOW,
    READ_TABLE_LAYOUT_RING,
    DsaMetadataBuilder,
    build_cache_specs_v41,
)


class V41PrefixMetadataTest(unittest.TestCase):
    def _builder(self, block_size: int = 128) -> DsaMetadataBuilder:
        caches, groups = build_cache_specs_v41([0, 2, 1], [1, 2], [1, 2], block_size, block_size)
        return DsaMetadataBuilder(caches, groups, read_table_layout=READ_TABLE_LAYOUT_RING)

    def _backend(self, window_size: int) -> Csa2AttentionBackend:
        return Csa2AttentionBackend(
            compress_ratios=[0, 2, 1],
            window_size=window_size,
            n_layers=3,
            num_heads=1,
            attn_head_dim=8,
            index_topk=8,
            index_n_heads=1,
            index_head_dim=8,
            rope_head_dim=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
            kv_source_layer_ids=[1, 2],
            index_source_layer_ids=[1, 2],
            candidate_source_layer_id=-1,
            candidate_topk_blocks=64,
            candidate_block_size=8,
            max_model_len=1024,
        )

    def test_bind_syncs_only_swa_descriptors_and_keeps_builder_references(self) -> None:
        for window in (128, 256, 512):
            with self.subTest(window=window):
                backend = self._backend(window)
                groups = backend.group_infos.copy()
                specs = [layer.copy() for layer in backend.caches_info]
                layer_caches = [LayerCache(key=None, value=None, swa=torch.zeros(12, 128, 8)) for _ in range(3)]
                backend.bind_kv_caches(layer_caches)
                self.assertIs(backend._kv_caches, layer_caches)
                self.assertIs(backend._builder.group_infos, backend.group_infos)
                self.assertIs(backend._builder.caches_info, backend.caches_info)
                self.assertEqual(backend.window_size, window)
                self.assertEqual(backend._swa_ring_blocks, window // 128)
                for original, bound in zip(groups, backend.group_infos, strict=True):
                    self.assertEqual(bound.block_size, 128)
                    self.assertEqual(bound.cache_type, original.cache_type)
                    self.assertEqual(bound.ratio, original.ratio)
                    if original.cache_type != DSA_CACHE_SLIDING_WINDOW or window == 128:
                        self.assertIs(bound, original)
                    else:
                        self.assertIsNot(bound, original)
                        self.assertEqual(original.block_size, window)
                for original_layer, bound_layer in zip(specs, backend.caches_info, strict=True):
                    for original, bound in zip(original_layer, bound_layer, strict=True):
                        self.assertEqual(bound.block_size, 128)
                        self.assertEqual(bound.group_id, original.group_id)
                        self.assertEqual(bound.cache_type, original.cache_type)
                        self.assertEqual(bound.ratio, original.ratio)
                        if original.cache_type != DSA_CACHE_SLIDING_WINDOW or window == 128:
                            self.assertIs(bound, original)
                        else:
                            self.assertIsNot(bound, original)
                            self.assertEqual(original.block_size, window)

                # Rebinding must derive the page width again, not hardcode 128.
                backend.bind_kv_caches([LayerCache(key=None, value=None, swa=torch.zeros(12, 64, 8))])
                self.assertEqual(backend._swa_ring_blocks, window // 64)
                for group in backend.group_infos:
                    self.assertEqual(group.block_size, 64 if group.cache_type == DSA_CACHE_SLIDING_WINDOW else 128)
                for original_layer, bound_layer in zip(specs, backend.caches_info, strict=True):
                    for original, bound in zip(original_layer, bound_layer, strict=True):
                        if original.cache_type == DSA_CACHE_SLIDING_WINDOW:
                            self.assertEqual(bound.block_size, 64)
                        else:
                            self.assertIs(bound, original)

    def test_unbound_swa_keeps_metadata_descriptors(self) -> None:
        backend = self._backend(512)
        groups = backend.group_infos.copy()
        specs = [layer.copy() for layer in backend.caches_info]
        for caches in ([], [LayerCache(key=None, value=None)]):
            backend.bind_kv_caches(caches)
            for original, bound in zip(groups, backend.group_infos, strict=True):
                self.assertIs(bound, original)
            for original_layer, bound_layer in zip(specs, backend.caches_info, strict=True):
                for original, bound in zip(original_layer, bound_layer, strict=True):
                    self.assertIs(bound, original)

    def test_bound_swa_suffix_slots_and_history_use_physical_pages(self) -> None:
        for window in (128, 256, 512):
            backend = self._backend(window)
            swa = torch.arange(32 * 128 * 8, dtype=torch.float32).view(32, 128, 8)
            backend.bind_kv_caches([LayerCache(key=None, value=None, swa=swa) for _ in range(3)])
            for prefix in (256, 512):
                for suffix in (3, 129, 257):
                    for graph_cols in (0, 8192):
                        with self.subTest(window=window, prefix=prefix, suffix=suffix, graph_cols=graph_cols):
                            total = prefix + suffix
                            count = (total + 127) // 128
                            raw = torch.full((1, count), -1, dtype=torch.int32)
                            shared_begin = max(0, prefix // 128 - window // 128)
                            raw[0, shared_begin : prefix // 128] = torch.arange(
                                9, 9 + prefix // 128 - shared_begin, dtype=torch.int32
                            )
                            raw[0, prefix // 128 :] = torch.arange(24, 24 + count - prefix // 128, dtype=torch.int32)
                            saved_table = raw.clone()
                            dsa = backend._builder.build(
                                multi_block_tables=[
                                    raw,
                                    torch.arange(1, 1 + (total + 255) // 256, dtype=torch.int32).view(1, -1),
                                    torch.arange(1, 1 + count, dtype=torch.int32).view(1, -1),
                                ],
                                kv_seq_lens=[total],
                                q_seq_lens=[suffix],
                                positions=torch.arange(prefix, total),
                                dsa_cos_sin=None,
                                is_prefill=True,
                                is_chunked_prefill=True,
                                enable_graph=graph_cols > 0,
                                graph_block_table_capacity_cols=graph_cols,
                            )
                            group = backend.swa_group_metadata(dsa)
                            self.assertIsNotNone(group)
                            self.assertIs(group.cache, swa)
                            self.assertEqual(group.block_size, 128)
                            positions = torch.arange(prefix, total)
                            expected = raw[0, positions // 128].to(torch.int64) * 128 + positions % 128
                            self.assertEqual(group.slot_mapping.tolist(), expected.tolist())
                            self.assertTrue(torch.equal(raw, saved_table))
                            history_positions = torch.arange(max(0, prefix - window), prefix)
                            history_slots = (
                                raw[0, history_positions // 128].to(torch.int64) * 128 + history_positions % 128
                            )
                            flat = swa.view(-1, 8)
                            history = flat[history_slots].clone()
                            shared_ids = raw[0, shared_begin : prefix // 128].to(torch.int64)
                            shared_before = swa[shared_ids].clone()
                            _scatter_by_slot(swa, group.slot_mapping, torch.full((suffix, 8), -13.0))
                            self.assertTrue(torch.equal(swa[shared_ids], shared_before))
                            self.assertTrue(torch.equal(flat[expected], torch.full((suffix, 8), -13.0)))
                            gathered = backend._gather_swa_positions(flat, group.block_table, 0, history_positions, 128)
                            self.assertTrue(torch.equal(gathered, history))

    def test_commits_only_new_c1_c2_rows_after_prefix(self) -> None:
        for prefix in (256, 512, 768):
            for suffix in (1, 2, 3, 127, 128, 129, 257):
                with self.subTest(prefix=prefix, suffix=suffix):
                    for ratio in (1, 2):
                        count = (prefix + suffix + 128 * ratio - 1) // (128 * ratio)
                        block_ids = torch.arange(20, 20 + count, dtype=torch.int32).view(1, -1)
                        saved = block_ids.clone()
                        _, slots = self._builder()._process_token_group(
                            block_ids, ratio, 128, [prefix + suffix], [suffix], 1, 0, 0
                        )
                        first = prefix // ratio
                        last = (prefix + suffix) // ratio
                        expected = [int(block_ids[0, row // 128]) * 128 + row % 128 for row in range(first, last)]
                        self.assertEqual(slots.tolist(), expected)
                        self.assertTrue(torch.equal(block_ids, saved))

    def test_restored_swa_tail_and_private_suffix_use_distinct_blocks(self) -> None:
        for prefix in (256, 512, 768):
            for suffix in (1, 2, 127, 128, 129):
                for graph_cols in (0, 8192):
                    with self.subTest(prefix=prefix, suffix=suffix, graph_cols=graph_cols):
                        logical_blocks = (prefix + suffix + 127) // 128
                        raw = torch.full((1, logical_blocks), -1, dtype=torch.int32)
                        restored_col = prefix // 128 - 1
                        raw[0, restored_col] = 9
                        for col in range(restored_col + 1, logical_blocks):
                            raw[0, col] = 10 + col - restored_col - 1
                        saved = raw.clone()
                        read_table, slots = self._builder()._process_swa_group(
                            raw,
                            128,
                            [prefix + suffix],
                            [suffix],
                            1,
                            suffix if graph_cols else 0,
                            graph_cols,
                            None,
                        )
                        expected = [int(raw[0, pos // 128]) * 128 + pos % 128 for pos in range(prefix, prefix + suffix)]
                        self.assertEqual(slots.tolist(), expected)
                        self.assertTrue(all(slot // 128 != 9 for slot in slots.tolist()))
                        visible_start = max(prefix, prefix + suffix - 128)
                        for pos in range(visible_start - 127, prefix + suffix):
                            if pos < prefix - 128:
                                continue
                            self.assertEqual(
                                int(read_table[0, (pos // 128) % read_table.size(1)]), int(raw[0, pos // 128])
                            )
                        self.assertTrue(torch.equal(raw, saved))

    def test_batch_lengths_do_not_mix_restored_swa_blocks(self) -> None:
        raw = torch.tensor([[-1, 9, 10, -1, -1], [-1, -1, -1, 11, 12]], dtype=torch.int32)
        read_table, slots = self._builder()._process_swa_group(
            raw,
            128,
            [257, 513],
            [1, 1],
            2,
            2,
            8192,
            None,
        )
        self.assertEqual(slots.tolist(), [10 * 128, 12 * 128])
        self.assertEqual(int(read_table[0, 1]), 9)
        self.assertEqual(int(read_table[1, 3]), 11)

    def test_graph_c2_parity_keeps_uncommitted_slot_invalid(self) -> None:
        raw = torch.tensor([[20, 21, 22]], dtype=torch.int32)
        for total in (257, 258, 259, 260):
            with self.subTest(total=total):
                _, slots = self._builder()._process_token_group(raw, 2, 128, [total], [1], 1, 1, 8)
                if total % 2:
                    self.assertEqual(slots.tolist(), [-1])
                else:
                    row = total // 2 - 1
                    self.assertEqual(slots.tolist(), [int(raw[0, row // 128]) * 128 + row % 128])

    def test_layer_caches_share_correct_restored_group_tables(self) -> None:
        builder = self._builder()
        tables = [
            torch.tensor([[-1, 9, 10]], dtype=torch.int32),
            torch.tensor([[20, 21]], dtype=torch.int32),
            torch.tensor([[30, 31, 32]], dtype=torch.int32),
        ]
        dsa = builder.build(
            multi_block_tables=tables,
            kv_seq_lens=[258],
            q_seq_lens=[2],
            positions=torch.tensor([256, 257], dtype=torch.int64),
            dsa_cos_sin=None,
            is_prefill=True,
            is_chunked_prefill=True,
        )
        self.assertEqual(dsa.slot_mappings[1][0].tolist(), [21 * 128])
        self.assertEqual(dsa.slot_mappings[2][0].tolist(), [32 * 128, 32 * 128 + 1])
        self.assertEqual(dsa.slot_mappings[1][2].tolist(), [10 * 128, 10 * 128 + 1])
        for idx in (2, 3, 4):
            self.assertEqual(dsa.block_tables[1][idx].data_ptr(), dsa.block_tables[0][0].data_ptr())
        self.assertEqual(dsa.block_tables[1][0].data_ptr(), dsa.block_tables[1][1].data_ptr())
        self.assertNotEqual(dsa.block_tables[1][0].data_ptr(), dsa.block_tables[2][0].data_ptr())


if __name__ == "__main__":
    unittest.main()
