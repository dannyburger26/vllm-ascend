# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/input_batch.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
from dataclasses import asdict, dataclass

import numpy as np
import torch
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.ops.rotary_embedding import update_cos_sin
from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num


class AscendInputBuffers(InputBuffers):
    """Input buffers for Ascend NPUs."""

    def __init__(
        self,
        max_num_reqs: int,
        max_num_tokens: int,
        device: torch.device,
    ):
        super().__init__(
            max_num_reqs,
            max_num_tokens,
            device,
        )
        del self.query_start_loc

        # NOTE: For FULL mode we change +1 to +2 to reserve extra space for padding.
        # See _pad_query_start_loc_for_fia.
        self.query_start_loc: torch.Tensor = torch.zeros(
            max_num_reqs + 2,
            dtype=torch.int32,
            device=device,
        )

        # Create seq_lens_cpu and seq_lens_np.
        # npu's attention backend still needs seq_lens on CPU side.
        self.seq_lens_cpu: torch.Tensor = torch.zeros(
            max_num_reqs,
            dtype=torch.int32,
            device="cpu",
        )
        # seq_len_np and seq_lens_cpu share the same memory.
        # define seq_lens_np for easier calculation with numpy.
        self.seq_lens_np: np.ndarray = self.seq_lens_cpu.numpy()

        # Logical KV coordinates for tree spec. RoPE keeps ``positions``
        # (depth-based); slot mapping uses this unique-per-token layout.
        self.slot_positions: torch.Tensor = torch.zeros(
            max_num_tokens,
            dtype=self.positions.dtype,
            device=device,
        )


@dataclass
class AscendInputBatch(InputBatch):
    """Input batch for Ascend NPUs."""

    # Create seq_lens_np.
    # npu's attention backend still needs seq_lens on CPU side.
    seq_lens_np: np.ndarray
    tree_visibility: torch.Tensor | None = None
    tree_num_nodes: torch.Tensor | None = None
    tree_tokens: torch.Tensor | None = None
    tree_depths: torch.Tensor | None = None
    tree_parents: torch.Tensor | None = None
    tree_first_child: torch.Tensor | None = None
    tree_next_sibling: torch.Tensor | None = None
    slot_positions: torch.Tensor | None = None
    path_node_ids: torch.Tensor | None = None
    # attn_state is used to build attention metadata.
    attn_state: AscendAttentionState | None = None

    @classmethod
    def make_dummy(
        cls,
        num_reqs: int,
        num_tokens: int,
        input_buffers: AscendInputBuffers,
    ) -> "AscendInputBatch":
        """Override the make_dummy method to calculate seq_lens_np."""
        input_batch = InputBatch.make_dummy(
            num_reqs,
            num_tokens,
            input_buffers,
        )
        # Evenly distribute num_tokens across requests instead of dumping the
        # whole remainder on the last request.
        # The old distribution could make the last dummy request's seq_len
        # exceed max_model_len, causing attention kernels to read block-table
        # entries past the tensor end (garbage page IDs / illegal memory access).
        base_tokens = num_tokens // num_reqs
        num_extra = num_tokens % num_reqs
        input_buffers.seq_lens_np[: num_reqs - num_extra] = base_tokens
        input_buffers.seq_lens_np[num_reqs - num_extra : num_reqs] = base_tokens + 1
        # Pad for full CUDA graph mode.
        input_buffers.seq_lens_np[num_reqs:] = 0
        seq_lens_np = input_buffers.seq_lens_np[:num_reqs]
        # A dummy run for dp or memory profiling.
        # When dummy run for dp, num_tokens is set to 1,
        # so attn_state is set to DecodeOnly.
        # when dummy run for memory profiling,
        # attention metadata isn't needed,
        # we can also set attn_state to AscendAttentionState.DecodeOnly.
        # For mla, update cos/sin. Here is for _dummy_run.
        update_cos_sin(input_batch.positions)
        return cls(
            **asdict(input_batch),
            seq_lens_np=seq_lens_np,
            attn_state=AscendAttentionState.DecodeOnly,
        )


@triton.jit
def _post_update_kernel(
    idx_mapping_ptr,
    idx_mapping_stride,
    num_computed_tokens_ptr,
    last_sampled_tokens_ptr,
    output_bin_counts_ptr,
    output_bin_counts_stride,
    sampled_tokens_ptr,
    sampled_tokens_stride,
    num_rows,
    num_sampled_ptr,
    num_rejected_ptr,
    query_start_loc_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    total_len_ptr,
):
    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)

    rows_per_program = (num_rows + n_programs - 1) // n_programs
    start_row = pid * rows_per_program
    end_row = tl.minimum(start_row + rows_per_program, num_rows)

    for row_idx in range(start_row, end_row):
        req_state_idx = tl.load(idx_mapping_ptr + row_idx * idx_mapping_stride)
        total_len = tl.load(total_len_ptr + req_state_idx)
        num_sampled = tl.load(num_sampled_ptr + row_idx)

        if num_sampled > 0:
            token_id = tl.load(sampled_tokens_ptr + row_idx * sampled_tokens_stride + num_sampled - 1)
            tl.store(last_sampled_tokens_ptr + req_state_idx, token_id)
            tl.store(total_len_ptr + req_state_idx, total_len + num_sampled)

        for i in range(num_sampled):
            token_id = tl.load(sampled_tokens_ptr + row_idx * sampled_tokens_stride + i)

            token_ptr = output_bin_counts_ptr + req_state_idx * output_bin_counts_stride + token_id
            count = tl.load(token_ptr)
            count += 1
            tl.store(token_ptr, count)

            tl.store(
                all_token_ids_ptr + req_state_idx * all_token_ids_stride + total_len + i,
                token_id,
            )

        query_start = tl.load(query_start_loc_ptr + row_idx)
        query_end = tl.load(query_start_loc_ptr + row_idx + 1)
        query_len = query_end - query_start
        num_rejected = tl.load(num_rejected_ptr + row_idx)

        num_computed = tl.load(num_computed_tokens_ptr + req_state_idx)
        num_computed += query_len - num_rejected
        tl.store(num_computed_tokens_ptr + req_state_idx, num_computed)


def post_update(
    # [num_reqs]
    idx_mapping: torch.Tensor,
    # [max_num_reqs]
    num_computed_tokens: torch.Tensor,
    # [max_num_reqs]
    last_sampled_tokens: torch.Tensor,
    # [max_num_reqs, vocab_size]
    output_bin_counts: torch.Tensor,
    # [num_reqs, num_speculative_steps + 1]
    sampled_tokens: torch.Tensor,
    # [num_reqs]
    num_sampled: torch.Tensor,
    # [num_reqs]
    num_rejected: torch.Tensor,
    # [num_reqs + 1]
    query_start_loc: torch.Tensor,
    # [max_num_reqs, max_model_len]
    all_token_ids: torch.Tensor,
    # [max_num_reqs]
    total_len: torch.Tensor,
) -> None:
    num_rows = idx_mapping.shape[0]

    core_num = get_vectorcore_num()

    grid = (min(num_rows, core_num),)
    _post_update_kernel[grid](
        idx_mapping,
        idx_mapping.stride(0),
        num_computed_tokens,
        last_sampled_tokens,
        output_bin_counts,
        output_bin_counts.stride(0),
        sampled_tokens,
        sampled_tokens.stride(0),
        num_rows,
        num_sampled,
        num_rejected,
        query_start_loc,
        all_token_ids,
        all_token_ids.stride(0),
        total_len,
    )


@triton.jit
def _prepare_tree_spec_pos_seq_lens_kernel(
    pos_ptr,
    slot_pos_ptr,
    seq_lens_ptr,
    idx_mapping_ptr,
    query_start_loc_ptr,
    is_prefilling_ptr,
    num_computed_tokens_ptr,
    tree_depths_ptr,
    tree_depths_stride,
    max_num_reqs,
    BLOCK_SIZE: tl.constexpr,
):
    req_id = tl.program_id(0)
    num_reqs = tl.num_programs(0) - 1
    if req_id == num_reqs:
        # Pad unused seq_lens as 0 for full CUDA graphs.
        for i in tl.range(num_reqs, max_num_reqs, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < max_num_reqs
            tl.store(seq_lens_ptr + block, 0, mask=mask)
        return

    req_state_idx = tl.load(idx_mapping_ptr + req_id)
    num_computed_tokens = tl.load(num_computed_tokens_ptr + req_state_idx)

    start = tl.load(query_start_loc_ptr + req_id)
    end = tl.load(query_start_loc_ptr + req_id + 1)
    query_len = end - start

    seq_len = num_computed_tokens + query_len
    tl.store(seq_lens_ptr + req_id, seq_len)

    is_prefill = tl.load(is_prefilling_ptr + req_id).to(tl.int32)
    if is_prefill == 1:
        for i in tl.range(0, query_len, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < query_len
            linear = num_computed_tokens + block
            tl.store(pos_ptr + start + block, linear, mask=mask)
            tl.store(slot_pos_ptr + start + block, linear, mask=mask)
        return

    # RoPE: root at num_computed; drafts use tree depth (siblings may share).
    # KV slots: unique token index so reshape_and_cache does not collide.
    tl.store(pos_ptr + start, num_computed_tokens)
    tl.store(slot_pos_ptr + start, num_computed_tokens)
    for i in tl.range(1, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        depth = tl.load(tree_depths_ptr + req_state_idx * tree_depths_stride + block - 1, mask=mask)
        rope_pos = depth + num_computed_tokens
        slot_pos = num_computed_tokens + block
        tl.store(pos_ptr + start + block, rope_pos, mask=mask)
        tl.store(slot_pos_ptr + start + block, slot_pos, mask=mask)


def prepare_tree_spec_pos_seq_lens(
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    is_prefilling: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    tree_depths: torch.Tensor,
    pos: torch.Tensor,
    slot_pos: torch.Tensor,
    seq_lens: torch.Tensor,
) -> None:
    """Fill RoPE ``pos`` and unique KV ``slot_pos`` on device."""
    num_reqs = idx_mapping.shape[0]
    _prepare_tree_spec_pos_seq_lens_kernel[(num_reqs + 1,)](
        pos,
        slot_pos,
        seq_lens,
        idx_mapping,
        query_start_loc,
        is_prefilling,
        num_computed_tokens,
        tree_depths,
        tree_depths.stride(0),
        seq_lens.shape[0],
        BLOCK_SIZE=1024,
    )
