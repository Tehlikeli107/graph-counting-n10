"""
Exhaustive n=9 Counting Revolution Proof
==========================================
Load all 274,668 non-isomorphic graphs on 9 vertices from McKay's database.
Compute induced 4-vertex subgraph signatures on GPU.
Check: does any pair of non-isomorphic graphs share the same signature?

If no collisions -> Counting Revolution covers n=9!
"""

import torch
import numpy as np
from itertools import combinations, permutations
import time
import os

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
GRAPH9_FILE = os.path.join(os.path.dirname(__file__), 'graph_data', 'graph9.g6')


def parse_graph6(line):
    line = line.strip()
    if line.startswith('>>graph6<<'):
        line = line[10:]
    data = [ord(c) - 63 for c in line]
    if data[0] <= 62:
        n = data[0]
        bits_start = 1
    else:
        n = ((data[1] & 63) << 12) | ((data[2] & 63) << 6) | (data[3] & 63)
        bits_start = 4
    bit_idx = 0
    A = np.zeros((n, n), dtype=np.int8)
    for j in range(1, n):
        for i in range(0, j):
            byte_pos = bits_start + bit_idx // 6
            bit_within = 5 - (bit_idx % 6)
            if byte_pos < len(data) and (data[byte_pos] >> bit_within) & 1:
                A[i, j] = A[j, i] = 1
            bit_idx += 1
    return A


def load_graphs(filename, max_graphs=None):
    graphs = []
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            if max_graphs and i >= max_graphs:
                break
            if line.strip():
                graphs.append(parse_graph6(line))
    return graphs


def precompute_k4_types():
    """Precompute canonical type for each 6-bit edge pattern (64 patterns -> 11 types)."""
    edges = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    canonical_reps = {}
    pattern_to_type = np.zeros(64, dtype=np.int32)
    type_id = 0

    for bits in range(64):
        A = np.zeros((4, 4), dtype=np.int8)
        for idx, (i, j) in enumerate(edges):
            if (bits >> idx) & 1:
                A[i, j] = A[j, i] = 1
        min_pattern = min(
            sum(int(A[perm[i], perm[j]]) << eidx
                for eidx, (i, j) in enumerate(edges))
            for perm in permutations(range(4))
        )
        if min_pattern not in canonical_reps:
            canonical_reps[min_pattern] = type_id
            type_id += 1
        pattern_to_type[bits] = canonical_reps[min_pattern]

    return pattern_to_type, type_id  # 11 types


def precompute_k5_types():
    """Precompute canonical type for each 10-bit edge pattern (1024 patterns -> 34 types)."""
    edges = [(i, j) for i in range(5) for j in range(i+1, 5)]
    canonical_reps = {}
    pattern_to_type = np.zeros(1024, dtype=np.int32)
    type_id = 0

    for bits in range(1024):
        A = np.zeros((5, 5), dtype=np.int8)
        for idx, (i, j) in enumerate(edges):
            if (bits >> idx) & 1:
                A[i, j] = A[j, i] = 1
        min_pattern = min(
            sum(int(A[perm[i], perm[j]]) << eidx
                for eidx, (i, j) in enumerate(edges))
            for perm in permutations(range(5))
        )
        if min_pattern not in canonical_reps:
            canonical_reps[min_pattern] = type_id
            type_id += 1
        pattern_to_type[bits] = canonical_reps[min_pattern]

    return pattern_to_type, type_id  # 34 types


def compute_signatures_gpu(graphs_np, k, type_lookup_np, num_types, batch_size=2048):
    """
    Compute induced k-subgraph type distributions for all graphs on GPU.
    graphs_np: (N, n, n) int8
    Returns: (N, num_types) int32
    """
    N, n, _ = graphs_np.shape
    subsets = list(combinations(range(n), k))
    num_subsets = len(subsets)

    if k == 4:
        edges = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    elif k == 5:
        edges = [(i,j) for i in range(5) for j in range(i+1,5)]
    ne = len(edges)

    lookup_t = torch.tensor(type_lookup_np, device=DEVICE, dtype=torch.int64)
    subsets_arr = np.array(subsets, dtype=np.int32)

    sigs = np.zeros((N, num_types), dtype=np.int32)

    for batch_start in range(0, N, batch_size):
        batch_end = min(batch_start + batch_size, N)
        B = batch_end - batch_start
        G_batch = torch.tensor(graphs_np[batch_start:batch_end],
                               device=DEVICE, dtype=torch.int64)  # (B, n, n)
        type_counts = torch.zeros(B, num_types, device=DEVICE, dtype=torch.int32)

        for si in range(num_subsets):
            subset = subsets_arr[si]
            # Compute edge pattern
            pattern = torch.zeros(B, device=DEVICE, dtype=torch.int64)
            for ei, (li, lj) in enumerate(edges):
                gi, gj = int(subset[li]), int(subset[lj])
                pattern = pattern | (G_batch[:, gi, gj] << ei)
            type_ids = lookup_t[pattern]  # (B,)
            type_counts.scatter_add_(
                1,
                type_ids.unsqueeze(1).to(torch.int64),
                torch.ones(B, 1, device=DEVICE, dtype=torch.int32)
            )

        sigs[batch_start:batch_end] = type_counts.cpu().numpy()

    return sigs


if __name__ == "__main__":
    print("="*65)
    print("EXHAUSTIVE n=9 COUNTING REVOLUTION PROOF")
    print("="*65)
    print(f"Device: {DEVICE}")
    if DEVICE == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Precompute type lookups
    print("Precomputing canonical type lookups...")
    t0 = time.time()
    k4_lookup, n4_types = precompute_k4_types()
    k5_lookup, n5_types = precompute_k5_types()
    print(f"  k=4: {n4_types} types, k=5: {n5_types} types  ({time.time()-t0:.1f}s)")
    print()

    # Load graphs
    print("Loading McKay n=9 catalog...")
    t0 = time.time()
    graphs = load_graphs(GRAPH9_FILE)
    N = len(graphs)
    print(f"  Loaded {N:,} graphs in {time.time()-t0:.1f}s")
    print()

    graphs_np = np.stack(graphs, axis=0)  # (N, 9, 9)

    # Degree sequences (fast baseline)
    deg = np.sort(graphs_np.sum(axis=2), axis=1)
    deg_unique = len(set(map(tuple, deg)))
    print(f"Degree sequences: {deg_unique:,} distinct / {N:,} total, "
          f"{N-deg_unique:,} collisions")

    # k=4 signatures
    print(f"\nComputing k=4 induced subgraph signatures (GPU, {N:,} graphs)...")
    t0 = time.time()
    sigs4 = compute_signatures_gpu(graphs_np, 4, k4_lookup, n4_types, batch_size=2048)
    t1 = time.time()
    sig4_set = set(map(tuple, sigs4))
    k4_unique = len(sig4_set)
    k4_coll = N - k4_unique
    print(f"  {n4_types} types, time: {t1-t0:.1f}s")
    print(f"  {k4_unique:,} distinct k=4 signatures")
    print(f"  Collisions: {k4_coll:,}")

    if k4_coll == 0:
        print()
        print("*** ZERO COLLISIONS with k=4! ***")
        print("Counting Revolution EXTENDS to n=9!")
    else:
        # k=5
        print(f"\nComputing k=5 induced subgraph signatures (GPU)...")
        t0 = time.time()
        sigs5 = compute_signatures_gpu(graphs_np, 5, k5_lookup, n5_types, batch_size=512)
        t1 = time.time()
        sig45 = set(map(tuple, np.concatenate([sigs4, sigs5], axis=1)))
        k45_unique = len(sig45)
        k45_coll = N - k45_unique
        print(f"  {n5_types} types, time: {t1-t0:.1f}s")
        print(f"  k=4+5 combined: {k45_unique:,} distinct")
        print(f"  Collisions: {k45_coll:,}")

        if k45_coll == 0:
            print()
            print("*** ZERO COLLISIONS with k<=5! ***")
            print("Counting Revolution EXTENDS to n=9!")
        else:
            # Show collision groups
            sig45_arr = [tuple(r) for r in np.concatenate([sigs4, sigs5], axis=1)]
            from collections import defaultdict
            groups = defaultdict(list)
            for i, s in enumerate(sig45_arr):
                groups[s].append(i)
            hard = [(s, idxs) for s, idxs in groups.items() if len(idxs) > 1]
            print(f"\n  {len(hard)} collision groups survive k<=5:")
            print("  These are candidates for quantum isomorphic pairs!")
            for _, idxs in hard[:3]:
                g1, g2 = graphs_np[idxs[0]], graphs_np[idxs[1]]
                e1, e2 = g1.sum()//2, g2.sum()//2
                print(f"    Graphs {idxs[0]},{idxs[1]}: {e1} edges, {e2} edges")

    print()
    print("="*65)
    print("RESULTS SUMMARY:")
    print(f"  n<=8: PROVEN complete (100% coverage)")
    print(f"  n=9:  {N:,} graphs tested")
    if k4_coll == 0:
        print(f"  n=9:  PROVEN complete with k=4 induced subgraphs")
    elif 'k45_coll' in dir() and k45_coll == 0:
        print(f"  n=9:  PROVEN complete with k<=5 induced subgraphs")
    else:
        num_qiso_candidates = len(hard)
        print(f"  n=9:  {num_qiso_candidates} potential quantum isomorphic pairs found!")
