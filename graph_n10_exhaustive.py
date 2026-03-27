"""
Exhaustive n=10 Counting Revolution Proof
==========================================
All 12,005,168 non-isomorphic graphs on 10 vertices.
Vectorized GPU computation: all subsets processed simultaneously.

Question: Does the Counting Revolution extend to n=10?
"""
import torch
import numpy as np
from itertools import combinations, permutations
import time
import os

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
GRAPH10_FILE = os.path.join(os.path.dirname(__file__), 'graph_data', 'graph10_decompressed.g6')


def fast_load_n10(filename):
    """Vectorized loading of all n=10 graphs from graph6."""
    with open(filename, 'rb') as f:
        raw = f.read()

    n = 10
    line_len = 1 + (n*(n-1)//2 + 5) // 6 + 1  # +1 for newline

    # For n=10: C(10,2)=45 bits, ceil(45/6)=8 chars -> 1+8+1=10 bytes per line
    line_len = 10
    N = len(raw) // line_len
    print(f"  Expected: {N:,} graphs (file={len(raw):,} bytes, line={line_len})")

    # Reshape: (N, 10)
    data = np.frombuffer(raw, dtype=np.uint8).reshape(N, line_len)
    # Columns 1..8 are edge data (column 0 = n encoding, column 9 = newline)
    edge_bytes = data[:, 1:9].astype(np.int32) - 63  # (N, 8) values in 0..63

    # Decode 45 bits from 8 6-bit groups
    # Bit positions: byte b, bit position 5-p within byte b
    # Edge (i,j) with i<j is edge number: sum_{k=0}^{j-1} k + i = j*(j-1)//2 + i
    edges = [(i, j) for j in range(1, n) for i in range(j)]  # 45 edges in graph6 order
    assert len(edges) == 45

    # Build adjacency matrices
    A = np.zeros((N, n, n), dtype=np.int8)
    for bit_idx, (i, j) in enumerate(edges):
        byte_pos = bit_idx // 6
        bit_within = 5 - (bit_idx % 6)
        bit = (edge_bytes[:, byte_pos] >> bit_within) & 1  # (N,)
        A[:, i, j] = bit
        A[:, j, i] = bit

    return A


def precompute_types(k):
    """Precompute canonical type for each edge bit pattern for k-vertex subgraph."""
    ek = [(i,j) for i in range(k) for j in range(i+1,k)]
    ne = len(ek)
    cr = {}; tid = 0
    lk = np.zeros(2**ne, dtype=np.int32)
    for bits in range(2**ne):
        A = np.zeros((k,k), dtype=np.int8)
        for idx,(i,j) in enumerate(ek):
            if (bits>>idx)&1: A[i,j]=A[j,i]=1
        mp = min(
            sum(int(A[p[i],p[j]])<<ei for ei,(i,j) in enumerate(ek))
            for p in permutations(range(k))
        )
        if mp not in cr: cr[mp]=tid; tid+=1
        lk[bits]=cr[mp]
    return lk, tid


def compute_sigs_vectorized(graphs_np, k, lookup_np, n_types, batch_size=1024):
    """
    Vectorized computation: ALL subsets processed simultaneously per batch.
    graphs_np: (N, n, n) int8
    Returns: (N, n_types) int32
    """
    N, n, _ = graphs_np.shape
    subsets = list(combinations(range(n), k))
    n_subs = len(subsets)
    ek = [(i,j) for i in range(k) for j in range(i+1,k)]
    ne = len(ek)

    # Precompute all (subset, edge) index pairs: (n_subs, ne, 2)
    sub_i = np.zeros((n_subs, ne), dtype=np.int32)
    sub_j = np.zeros((n_subs, ne), dtype=np.int32)
    for si, sub in enumerate(subsets):
        for ei, (li, lj) in enumerate(ek):
            sub_i[si, ei] = sub[li]
            sub_j[si, ei] = sub[lj]

    # Move index arrays to GPU
    sub_i_t = torch.tensor(sub_i, device=DEVICE, dtype=torch.int64)  # (n_subs, ne)
    sub_j_t = torch.tensor(sub_j, device=DEVICE, dtype=torch.int64)  # (n_subs, ne)
    lookup_t = torch.tensor(lookup_np, device=DEVICE, dtype=torch.int64)
    # Bit multipliers: 1, 2, 4, 8, ...
    bit_mults = torch.tensor([1 << i for i in range(ne)], device=DEVICE, dtype=torch.int64)  # (ne,)

    sigs = np.zeros((N, n_types), dtype=np.int32)

    for b0 in range(0, N, batch_size):
        b1 = min(b0 + batch_size, N)
        B = b1 - b0

        G = torch.tensor(graphs_np[b0:b1], device=DEVICE, dtype=torch.int64)  # (B, n, n)

        # Extract all edge values for all subsets at once
        # G[:, sub_i_t, sub_j_t]: index G with (n_subs, ne) index pairs
        # Result: (B, n_subs, ne)
        edges_vals = G[:, sub_i_t, sub_j_t]  # (B, n_subs, ne)

        # Compute bit patterns: (B, n_subs) int64
        patterns = (edges_vals * bit_mults.unsqueeze(0).unsqueeze(0)).sum(dim=2)  # (B, n_subs)

        # Lookup types: (B, n_subs)
        type_ids = lookup_t[patterns]  # (B, n_subs)

        # Count types per graph: (B, n_types)
        counts = torch.zeros(B, n_types, device=DEVICE, dtype=torch.int32)
        for t in range(n_types):
            counts[:, t] = (type_ids == t).sum(dim=1)

        sigs[b0:b1] = counts.cpu().numpy()

    return sigs


if __name__ == "__main__":
    print("="*65)
    print("EXHAUSTIVE n=10 COUNTING REVOLUTION PROOF")
    print("="*65)
    print(f"Device: {DEVICE}")
    if DEVICE == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Precompute type lookups
    print("Precomputing type lookups...")
    t0 = time.time()
    k4_lk, n4t = precompute_types(4)
    k5_lk, n5t = precompute_types(5)
    t1 = time.time()
    print(f"  k=4: {n4t} types, k=5: {n5t} types ({t1-t0:.1f}s)")

    # Load graphs
    print("\nLoading n=10 graphs (vectorized)...")
    t0 = time.time()
    graphs_np = fast_load_n10(GRAPH10_FILE)
    N = len(graphs_np)
    t1 = time.time()
    print(f"  Loaded {N:,} graphs in {t1-t0:.1f}s")
    print(f"  Memory: {graphs_np.nbytes/1e9:.2f} GB")
    print()

    # k=4 signatures (vectorized)
    print(f"Computing k=4 signatures ({N:,} graphs, C(10,4)={len(list(combinations(range(10),4)))} subsets)...")
    t0 = time.time()
    sigs4 = compute_sigs_vectorized(graphs_np, 4, k4_lk, n4t, batch_size=512)
    t1 = time.time()
    sig4_set = set(map(tuple, sigs4))
    k4_coll = N - len(sig4_set)
    print(f"  Time: {t1-t0:.1f}s")
    print(f"  {len(sig4_set):,} distinct, {k4_coll:,} collisions")

    if k4_coll == 0:
        print("*** ZERO COLLISIONS with k=4! Counting Revolution extends to n=10! ***")
    else:
        # k=5
        print(f"\nComputing k=5 signatures...")
        t0 = time.time()
        sigs5 = compute_sigs_vectorized(graphs_np, 5, k5_lk, n5t, batch_size=128)
        t1 = time.time()
        combined = np.concatenate([sigs4, sigs5], axis=1)
        sig45_set = set(map(tuple, combined))
        k45_coll = N - len(sig45_set)
        print(f"  Time: {t1-t0:.1f}s")
        print(f"  k=4+5: {len(sig45_set):,} distinct, {k45_coll:,} collisions")

        if k45_coll == 0:
            print("*** ZERO COLLISIONS with k<=5! Counting Revolution extends to n=10! ***")
        else:
            print(f"  Need k=6 for remaining {k45_coll} collisions")
            # (k=6 precomputation takes ~55s, add if needed)

    print()
    print("="*65)
    print("SUMMARY:")
    print(f"  n=7:  1,044 graphs  - PROVEN (k<=4)")
    print(f"  n=8:  12,346 graphs - PROVEN (k<=4 or 5)")
    print(f"  n=9:  274,668 graphs - PROVEN (k<=6)")
    print(f"  n=10: {N:,} graphs - {'PROVEN' if k4_coll==0 else ('PROVEN (k<=5)' if 'k45_coll' in dir() and k45_coll==0 else 'IN PROGRESS')}")
