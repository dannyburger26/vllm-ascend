"""Eager tree verification with private cache tensors per layer.

Persistent caches are read during verification. Only accepted branch snapshots
are copied back. Block 0 remains the null block in every temporary cache.
"""
from __future__ import annotations

from contextlib import contextmanager

import torch

_LAYOUT: dict | None = None
_PLAN: dict | None = None
_KV_STACK: list[list[torch.Tensor]] = []


def _key(tensor):
    return (tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype)


def set_tree_chain_layout(parents, num_nodes, depths, query_start_loc, start_pos,
                          idx_mapping=None) -> None:
    global _LAYOUT, _PLAN
    if _PLAN is not None and not _PLAN.get("committed", False):
        raise RuntimeError("Previous tree verification was not committed")
    _LAYOUT = dict(parents=parents, num_nodes=num_nodes, depths=depths,
                   qsl=query_start_loc, start=start_pos, idx=idx_mapping)
    _PLAN = None


def clear_tree_chain_layout() -> None:
    global _LAYOUT, _PLAN
    if _PLAN is not None and not _PLAN.get("committed", False):
        raise RuntimeError("Cannot discard uncommitted tree state")
    _LAYOUT = None
    _PLAN = None


def chain_layout_active() -> bool:
    return _LAYOUT is not None


@contextmanager
def chain_kv_scope(caches):
    _KV_STACK.append([cache for cache in caches if cache is not None])
    try:
        yield
    finally:
        _KV_STACK.pop()


def _build_plan(num_tokens: int, device) -> dict | None:
    global _PLAN
    if _PLAN is not None:
        return _PLAN
    if _LAYOUT is None:
        return None
    layout = _LAYOUT
    n_req = int(layout["num_nodes"].numel())
    if n_req == 0:
        raise RuntimeError("Empty tree verification batch")
    # D2H: build the host branch plan once per verification, before all layers.
    qsl = layout["qsl"][:n_req + 1].detach().cpu().tolist()
    prefixes = layout["start"][:n_req].detach().cpu().tolist()
    counts = layout["num_nodes"].detach().cpu().tolist()
    parents = layout["parents"][:n_req].detach().cpu().tolist()
    depths = layout["depths"][:n_req].detach().cpu().tolist()
    if len(qsl) != n_req + 1 or qsl[0] != 0 or qsl[-1] > num_tokens:
        raise RuntimeError("Invalid tree query boundaries")
    rows, cu, chains, req_index, prefix_of, token_index = [], [0], [], [], [], []
    req_cu = [0]
    for req in range(n_req):
        begin, end = int(qsl[req]), int(qsl[req + 1])
        proposed_count = int(counts[req])
        if not 0 <= proposed_count <= len(parents[req]) or end <= begin:
            raise RuntimeError("Invalid tree node count/query length")
        # Near the output limit, only the scheduled prefix of the flat tree
        # exists. The beam layout is topological, so that prefix is closed.
        count = min(proposed_count, end - begin - 1)
        for local in range(end - begin):
            chain = []
            if 0 < local <= count:
                node, visited = local, set()
                while node:
                    if node in visited or not 1 <= node <= count:
                        raise RuntimeError("Invalid/cyclic tree parent links")
                    visited.add(node)
                    chain.append(node)
                    node = int(parents[req][node - 1])
                chain.reverse()
                if len(chain) != int(depths[req][local - 1]):
                    raise RuntimeError("Tree depth does not match its ancestor path")
            # Root is a real target input at prefix P, including when A == 0.
            rows.extend([begin] + [begin + node for node in chain])
            cu.append(len(rows))
            chains.append(chain)
            req_index.append(req)
            prefix_of.append(int(prefixes[req]))
            token_index.append(begin + local)
        req_cu.append(len(chains))
    lens = [len(chain) + 1 for chain in chains]
    seq_lens = [p + n for p, n in zip(prefix_of, lens)]
    idx = layout["idx"]
    _PLAN = {
        "n": len(chains), "n_req": n_req, "req_cu": req_cu,
        "idx": None if idx is None else idx[:n_req].detach().cpu().tolist(),
        "req_index": req_index, "prefixes": prefix_of, "chains": chains,
        "lens": lens, "max_q": max(lens), "max_kv": max(seq_lens),
        "token_count": cu[-1],
        "rows": torch.tensor(rows, dtype=torch.long, device=device),
        "cu": torch.tensor(cu, dtype=torch.int32, device=device),
        "end_rows": torch.tensor([end - 1 for end in cu[1:]],
                                 dtype=torch.long, device=device),
        "start": torch.tensor(prefix_of, dtype=torch.int32, device=device),
        "seqused": torch.tensor(seq_lens, dtype=torch.int32, device=device),
        "token_index": torch.tensor(token_index, dtype=torch.long, device=device),
        "cache_views": {}, "kv_tables": {}, "ori_views": {}, "index_topk": {},
        "length_cus": {}, "indexer_lengths": None,
        "pending": [[] for _ in chains], "records": [[] for _ in chains],
        "committed": False,
    }
    return _PLAN


def _new_cache_views(caches, num_pages):
    """Preserve aliases such as packed FP8 key/scale/full-cache views."""
    groups, result = [], [None] * len(caches)
    order = sorted(range(len(caches)), key=lambda i: caches[i].data_ptr())
    for i in order:
        cache = caches[i]
        if cache.ndim < 2 or any(stride < 0 for stride in cache.stride()):
            raise RuntimeError("Unsupported cache tensor strides")
        size = cache.element_size()
        step = int(cache.stride(0)) * size
        span = size + sum((int(n) - 1) * int(s) * size
                          for n, s in zip(cache.shape[1:], cache.stride()[1:]))
        if step <= 0 or span > step:
            raise RuntimeError("Overlapping/invalid physical cache pages")
        storage = cache.untyped_storage().data_ptr()
        group = next((g for g in groups if g["storage"] == storage
                      and g["step"] == step
                      and 0 <= cache.data_ptr() - g["base"]
                      and cache.data_ptr() - g["base"] + span <= step), None)
        if group is None:
            group = {"storage": storage, "step": step, "base": cache.data_ptr(),
                     "bytes": torch.zeros(num_pages * step, dtype=torch.uint8,
                                          device=cache.device)}
            groups.append(group)
        offset = cache.data_ptr() - group["base"]
        if offset % size:
            raise RuntimeError("Misaligned packed-cache view")
        result[i] = torch.empty(0, dtype=cache.dtype, device=cache.device).set_(
            group["bytes"].untyped_storage(), offset // size,
            (num_pages, *cache.shape[1:]), tuple(cache.stride()),
        )
    return result


def _workspace(plan, caches, block_table, real, write_cols, kind, read_cols=None):
    """Copy shared read pages once; allocate distinct write pages per chain.

    No temporary block ID ever refers to storage in a persistent cache.
    Aliased block-table entries within one chain retain their aliasing.
    ``real`` is this call's host snapshot of the block table; return the host
    temporary rows too so callers need not copy their new device table back.
    """
    width = int(block_table.shape[1])
    rows = [[0] * width for _ in range(plan["n"])]
    sources = [0]  # Private, zero-filled null page.
    allocated = {}
    private = [[] for _ in rows]
    for chain, req in enumerate(plan["req_index"]):
        if not 0 <= req < len(real):
            raise RuntimeError("Missing request block-table row")
        original = real[req]
        columns = range(width) if read_cols is None else read_cols[chain]
        writable = set()
        for col in write_cols[chain]:
            if not 0 <= col < width or int(original[col]) <= 0:
                raise RuntimeError("Tree write targets an unallocated/null cache page")
            writable.add(int(original[col]))
        for col in set(columns) | set(write_cols[chain]):
            if not 0 <= col < width:
                raise RuntimeError("Tree cache column exceeds block-table capacity")
            src = int(original[col])
            if src <= 0:
                continue
            owner = chain if src in writable else -1
            tag = (owner, src)
            if tag not in allocated:
                allocated[tag] = len(sources)
                sources.append(src)
                if owner >= 0:
                    private[chain].append((src, allocated[tag]))
            rows[chain][col] = allocated[tag]
    table = torch.tensor(rows, dtype=torch.int32, device=block_table.device)
    views, seen = _new_cache_views(caches, len(sources)), set()
    max_source = max(sources)
    # H2D: all aliased cache views use the same page selection.
    source_ids = torch.tensor(sources[1:], dtype=torch.long, device=caches[0].device)
    for cache, view in zip(caches, views):
        key = _key(cache)
        if key in seen or key in plan["cache_views"]:
            raise RuntimeError("Duplicate cache binding/compressor call in one tree layer")
        seen.add(key)
        if cache.ndim < 2 or max_source >= int(cache.shape[0]):
            raise RuntimeError("Cache tensor does not match its block table")
        if len(sources) > 1:
            view[1:].copy_(cache.index_select(0, source_ids))
        for binding in (cache, view):
            plan["cache_views"][_key(binding)] = view
            plan["kv_tables"][_key(binding)] = table
        if kind != "ori":
            for chain, pages in enumerate(private):
                for src, dst in pages:
                    plan["pending"][chain].append((cache, src, view, dst, kind))
    return views, table, rows


def tree_cache_view(cache):
    """Resolve the cache matching temporary compressor slot IDs/block tables."""
    if cache is None or _LAYOUT is None:
        return cache
    if _PLAN is None or _key(cache) not in _PLAN["cache_views"]:
        raise RuntimeError("Tree cache writer/reader has no private workspace")
    return _PLAN["cache_views"][_key(cache)]


def lookup_kv_table(cache):
    if _LAYOUT is None:
        return None
    if _PLAN is None or _key(cache) not in _PLAN["kv_tables"]:
        raise RuntimeError("Tree attention/indexer has no private block table")
    return _PLAN["kv_tables"][_key(cache)]


def _emit_count(start, length, ratio):
    return (start + length) // ratio - start // ratio


def chain_compressor_batch(hidden, state_cache, kv_block_table, state_block_table,
                           storage_block_size, ratio, req_metadata, slot_format):
    if _LAYOUT is None:
        return None
    if not _KV_STACK or not _KV_STACK[-1]:
        raise RuntimeError("Tree compressor called without its output cache bindings")
    plan = _build_plan(int(hidden.shape[0]), hidden.device)
    ratio, bs = int(ratio), int(storage_block_size)
    if ratio not in (4, 128) or bs <= 0:
        raise RuntimeError("Unsupported tree compression layout")
    caches = _KV_STACK[-1]
    if int(caches[0].shape[1]) != bs:
        raise RuntimeError("Compressed cache page size differs from metadata")
    writes, reads = [], []
    for prefix, length in zip(plan["prefixes"], plan["lens"]):
        begin, end = prefix // ratio, (prefix + length) // ratio
        writes.append(sorted({position // bs for position in range(begin, end)}))
        reads.append(list(range((end + bs - 1) // bs)))
    # D2H: snapshot each table at its layer use; metadata storage can be reused.
    real_kv = kv_block_table.detach().cpu().tolist()
    _, kv_rows, _ = _workspace(
        plan, caches, kv_block_table, real_kv, writes, "kv", reads
    )

    # State caches can reuse physical pages through circular/aliased tables.
    # Copy all referenced state pages privately and preserve those aliases.
    real_state = state_block_table.detach().cpu().tolist()  # D2H
    state_writes = [
        [col for col, block in enumerate(real_state[req]) if int(block) > 0]
        for req in plan["req_index"]
    ]
    state_views, state_rows, _ = _workspace(
        plan, [state_cache], state_block_table, real_state, state_writes, "state"
    )
    full_cos, full_sin = req_metadata.full_compress_cos, req_metadata.full_compress_sin
    cos = full_cos.view(full_cos.shape[0], full_cos.shape[-1])
    sin = full_sin.view(full_sin.shape[0], full_sin.shape[-1])
    token_size = plan["token_count"]
    n_rows = max(1, min(token_size, token_size // ratio + plan["n"]))
    compress_cos, compress_sin, slots = torch.ops._C_ascend.compressor_metadata(
        cos, sin, plan["cu"], plan["start"], kv_rows, bs, int(slot_format),
        ratio, n_rows, plan["n"],
    )
    return {
        "hidden": hidden.index_select(0, plan["rows"]),
        "cu": plan["cu"], "start": plan["start"], "kv_bt": kv_rows,
        "state_bt": state_rows, "state": state_views[0],
        "cos": compress_cos, "sin": compress_sin, "slot": slots,
    }


def pack_chain_tokens(tensor):
    if tensor is None or _PLAN is None:
        return tensor
    return tensor.index_select(0, _PLAN["rows"])


def unpack_chain_tokens(packed, n_tokens):
    if _PLAN is None:
        return packed
    ends = packed.index_select(0, _PLAN["end_rows"])
    out = packed.new_zeros((n_tokens, *ends.shape[1:]))
    out.index_copy_(0, _PLAN["token_index"], ends)
    return out


def chain_length_cus(seqused, cmp_ratio):
    # Only plan-owned lengths are immutable across layers. Other callers keep
    # the standalone calculation, with no pointer-based content cache.
    cached = _PLAN is not None and seqused is _PLAN["seqused"]
    if cached and cmp_ratio in _PLAN["length_cus"]:
        return _PLAN["length_cus"][cmp_ratio]
    ori = seqused.new_zeros(seqused.numel() + 1)
    cmp = seqused.new_zeros(seqused.numel() + 1)
    ori[1:] = torch.cumsum(seqused, dim=0)
    cmp[1:] = torch.cumsum(torch.div(seqused, cmp_ratio, rounding_mode="floor"), dim=0)
    if cached:
        _PLAN["length_cus"][cmp_ratio] = (ori, cmp)
    return ori, cmp


def chain_attention_view(ori_cache, ori_block_table, block_size, window_size):
    if _LAYOUT is None:
        return None
    if _PLAN is None:
        raise RuntimeError("Tree attention ran before branch compression")
    plan = _PLAN
    key = _key(ori_cache)
    if key in plan["ori_views"]:
        return plan["ori_views"][key]
    bs = int(block_size)
    if int(ori_cache.shape[1]) != bs or int(window_size) <= 0:
        raise RuntimeError("Invalid original-KV page/window size")
    writes, reads = [], []
    for prefix, chain in zip(plan["prefixes"], plan["chains"]):
        writes.append(sorted({(prefix + depth) // bs
                              for depth in range(1, len(chain) + 1)}))
        lo, hi = max(0, prefix - int(window_size) + 1), prefix + len(chain)
        reads.append(list(range(lo // bs, hi // bs + 1)))
    real = ori_block_table.detach().cpu().tolist()  # D2H: one snapshot for this layer.
    views, table, temporary = _workspace(
        plan, [ori_cache], ori_block_table, real, writes, "ori", reads
    )
    work = views[0]
    copies = {}
    for endpoint, chain in enumerate(plan["chains"]):
        prefix, req = plan["prefixes"][endpoint], plan["req_index"][endpoint]
        for depth, node in enumerate(chain, 1):
            source_pos, dest_pos = prefix + node, prefix + depth
            source_col, dest_col = source_pos // bs, dest_pos // bs
            if source_col >= len(real[req]):
                raise RuntimeError("Tree source slot exceeds block-table capacity")
            src, dst = int(real[req][source_col]), int(temporary[endpoint][dest_col])
            if src <= 0 or dst <= 0:
                raise RuntimeError("Tree original-KV copy references a null page")
            # Circular tables may alias a destination slot. Match the original
            # ordered copies by retaining its last source before the batch copy.
            copies[(dst, dest_pos % bs)] = (src, source_pos % bs)
    if copies:
        # H2D: one index batch; keep cache views/strides and byte values intact.
        indices = torch.tensor(
            [(*dst, *src) for dst, src in copies.items()],
            dtype=torch.long, device=work.device,
        ).unbind(1)
        source, target = ori_cache, work
        if source.element_size() == 1:
            # Transport FP8 payloads as bytes without requiring FP8 index_put.
            source, target = source.view(torch.uint8), target.view(torch.uint8)
        target[indices[0], indices[1]] = source[indices[2], indices[3]]
    view = {"cu": plan["cu"], "seqused": plan["seqused"],
            "max_q": plan["max_q"], "max_kv": plan["max_kv"],
            "ori_bt": table, "ori_cache": work}
    plan["ori_views"][key] = view
    return view


def indexer_batch(key_cache):
    if _LAYOUT is None:
        return None
    table = lookup_kv_table(key_cache)
    plan = _PLAN
    seq = plan["seqused"]
    if plan["indexer_lengths"] is None:
        plan["indexer_lengths"] = (
            torch.div(seq, 4, rounding_mode="floor"), torch.remainder(seq, 4)
        )
    seqused_k, residual = plan["indexer_lengths"]
    return {"cu": plan["cu"], "seqused_k": seqused_k,
            "residual": residual, "block_table": table,
            "max_q": plan["max_q"], "max_k": plan["max_kv"] // 4,
            "batch": plan["n"]}


def cache_tree_topk(buffer, indices, offset=0):
    if _PLAN is None or buffer is None or offset != 0:
        raise RuntimeError("Invalid packed TopK cache write")
    if int(indices.shape[0]) != _PLAN["token_count"]:
        raise RuntimeError("TopK output does not use packed chain rows")
    # The indexer owns a fresh output tensor; subsequent layers only read it.
    _PLAN["index_topk"][_key(buffer)] = indices


def get_tree_topk(buffer, offset=0):
    if _PLAN is None or buffer is None or offset != 0:
        raise RuntimeError("Invalid packed TopK cache read")
    indices = _PLAN["index_topk"].get(_key(buffer))
    if indices is None:
        raise RuntimeError("No TopK producer for this tree verification step")
    return indices


def finish_tree_layer() -> None:
    """Retain compact writable snapshots; release copied prefix pages.

    State-page dirtiness is compared byte-for-byte on device. Only the small
    dirty mask crosses to the host, once per layer, to discard inactive pages
    before retaining snapshots for the later winner commitment.
    """
    if _PLAN is None:
        return
    if _KV_STACK:
        raise RuntimeError("Tree layer finished inside a compressor scope")
    groups = {}
    records = [[None] * len(pending) for pending in _PLAN["pending"]]
    for chain, pending in enumerate(_PLAN["pending"]):
        for pos, (original, dst, workspace, src, kind) in enumerate(pending):
            key = (_key(original), _key(workspace), kind)
            if key not in groups:
                groups[key] = (original, workspace, kind, [])
            groups[key][3].append((chain, pos, dst, src))

    dirty_masks = []
    for original, workspace, kind, entries in groups.values():
        if kind != "state":
            continue
        # H2D: batch page indices instead of a separate compare per page.
        indices = torch.tensor(
            [(dst, src) for _, _, dst, src in entries],
            dtype=torch.long, device=original.device,
        ).unbind(1)
        before = original.index_select(0, indices[0]).contiguous().view(torch.uint8)
        after = workspace.index_select(0, indices[1]).contiguous().view(torch.uint8)
        dirty_masks.append((before != after).reshape(len(entries), -1).any(dim=1))
        del before, after
    # D2H: one host filtering decision per layer, replacing torch.equal per page.
    dirty = torch.cat(dirty_masks).cpu().tolist() if dirty_masks else []
    offset = 0
    for original, workspace, kind, entries in groups.values():
        if kind == "state":
            changed = dirty[offset : offset + len(entries)]
            offset += len(entries)
            # Full state tables can retain inactive aliases. Do not commit an
            # unchanged page over another request's winning update.
            entries = [entry for entry, keep in zip(entries, changed) if keep]
        if not entries:
            continue
        source_ids = torch.tensor(
            [src for _, _, _, src in entries], dtype=torch.long, device=workspace.device
        )  # H2D
        snapshot = workspace.index_select(0, source_ids)
        for row, (chain, pos, dst, _) in enumerate(entries):
            records[chain][pos] = (original, dst, snapshot[row], kind)

    for chain, pending in enumerate(_PLAN["pending"]):
        # Preserve the original order, including overlapping packed-cache views.
        _PLAN["records"][chain].extend(
            record for record in records[chain] if record is not None
        )
        pending.clear()
    _PLAN["cache_views"].clear()
    _PLAN["kv_tables"].clear()
    _PLAN["ori_views"].clear()


def commit_winner(path_node_ids, num_sampled=None, idx_mapping=None):
    if _PLAN is None or path_node_ids is None:
        raise RuntimeError("Missing tree verification plan/accepted path")
    plan = _PLAN
    if plan["committed"] or any(plan["pending"]):
        raise RuntimeError("Tree state committed twice or a layer was not finalized")
    paths = path_node_ids.detach().cpu().tolist()
    if len(paths) != plan["n_req"]:
        raise RuntimeError("Accepted-path batch differs from verification batch")
    if plan["idx"] is not None:
        if idx_mapping is None or idx_mapping[:plan["n_req"]].detach().cpu().tolist() != plan["idx"]:
            raise RuntimeError("Request order changed before tree state commitment")
    sampled = None if num_sampled is None else num_sampled.detach().cpu().tolist()
    winners = []
    for req, row in enumerate(paths):
        if sampled is not None and int(sampled[req]) == 0:
            continue
        accepted, ended = [], False
        for value in row:
            node = int(value)
            if node == -1:
                ended = True
            elif ended or node <= 0:
                raise RuntimeError("Invalid accepted-path padding/node ID")
            else:
                accepted.append(node)
        if sampled is not None and int(sampled[req]) != len(accepted) + 1:
            raise RuntimeError("Sample count does not match accepted path plus bonus")
        winner = plan["req_cu"][req] + (accepted[-1] if accepted else 0)
        if winner >= plan["req_cu"][req + 1] or plan["chains"][winner] != accepted:
            raise RuntimeError("Accepted nodes are not a verified root-to-node path")
        winners.append(winner)
    # Validate shared destinations before mutating any persistent cache.
    copies = {}
    for winner in winners:
        for cache, page, value, kind in plan["records"][winner]:
            key = (_key(cache), page)
            if key in copies and not torch.equal(copies[key][2], value):
                raise RuntimeError("Two requests commit different data to one persistent page")
            copies[key] = (cache, page, value)
    for cache, page, value in copies.values():
        cache[page].copy_(value)
    plan["committed"] = True
    # Release rejected branch snapshots before the next proposal.
    for records in plan["records"]:
        records.clear()
    return {"commit": 1, "requests": len(winners), "pages": len(copies)}
