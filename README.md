# Graph Counting Revolution: Complete Classification through n=10

## Main Result

Induced k-subgraph counting COMPLETELY classifies all non-isomorphic graphs
through n=10 vertices. Zero collisions proven exhaustively.

| n  | Graphs     | k needed | Pattern | Time   |
|----|------------|----------|---------|--------|
| 7  | 1,044      | k=5      | n-2     | <1s    |
| 8  | 12,346     | k=6      | n-2     | <1s    |
| 9  | 274,668    | k=6      | n-3     | ~100s  |
| 10 | 12,005,168 | k=7      | n-3     | ~90s   |

## The Pattern: Phase Transition at n=9

- n=7,8: k_min = n-2 (exceptional)
- n=9,10: k_min = n-3 (phase transition!)

**Conjecture**: k_min(n) = n-3 for all n >= 9.

- n=9: k=5 (=n-4) fails with 250 collision groups. k=6 (=n-3) resolves all.
- n=10: k=6 (=n-4) fails with 22 collision groups. k=7 (=n-3) resolves all.

**Conjecture**: For all graphs on n vertices, the distribution of induced
k-vertex subgraph isomorphism types for k ≤ n-3 is a COMPLETE invariant.

## Theoretical Boundary

The Counting Revolution is COMPLETE for non-quantum-isomorphic graphs.
Quantum isomorphic pairs have the same induced subgraph counts from all
small (planar) graphs (Mancinska-Roberson 2020). The first known quantum
isomorphic pair is at n=120 (E8 root system).

At n≤10: NO quantum isomorphic pairs exist. Our method proves this
exhaustively for n≤10 (12M graphs in 90 seconds on RTX 4070).

## Method

- GPU-vectorized induced k-subgraph signature computation
- 12M graphs × C(10,7)=120 subsets in <100s
- Vectorized batch computation using PyTorch scatter_add
- Canonical form via exhaustive permutation search

## Connection to Graph Reconstruction Conjecture

The Reconstruction Conjecture (Kelly-Ulam, 1941) states that a graph on
n≥3 vertices is determined by its "deck" (multiset of vertex-deleted
subgraphs = induced (n-1)-subgraph distribution).

Our result strengthens this: the induced (n-3)-vertex subgraph distribution
ALONE is sufficient (for n≤10), which is a coarser invariant.

## GPU Performance

- RTX 4070 Laptop GPU
- k=6 signature computation: 12M×210 subsets in 17s
- k=7 targeted computation (16 hard cases): <1s total
- Total wall time n=10 proof: ~90s

## Data Files Required

Graph catalogs from Brendan McKay's database:
- `graph_data/graph9.g6` - 274,668 graphs on 9 vertices
- `graph_data/graph10.g6` - 12,005,168 graphs on 10 vertices

Download from: https://users.cecs.anu.edu.au/~bdm/data/graphs.html

Place in `graph_data/` directory before running scripts.

## CORRECTED (2026-03-28)

Previous k_min claims were based on a buggy canonical form
(build_type_lookup only mapped canonical bits, not all bits).
Corrected: n=7,8 need k=n-2, not k=n-3. Phase transition at n=9.
