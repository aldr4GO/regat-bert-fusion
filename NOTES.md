# NOTES — deviations from the plan, VERIFY outcomes, results summary

Paper numbers (ReGAT-BERT, MS MARCO dev: MRR@10 0.413, NDCG@10 0.794) are **reported, protocol unknown**;
they are never mixed into our tables or significance tests (Sec. 3.0). The paper does not state split sizes,
candidate depth, negative sampling, graph construction on MS MARCO, GAT hyperparameters, or how Sim_sem is
bounded for the BCE; every comparison row is reproduced here under one protocol.

## VERIFY outcomes

(filled in during implementation / testing — see bottom of each item)

- **ir_datasets ids**: `msmarco-passage/train/judged`, `msmarco-passage/dev/small`,
  `msmarco-passage/trec-dl-2019/judged`, `msmarco-passage/trec-dl-2020/judged`, collection `msmarco-passage`.
- **BM25**: Pyserini prebuilt `msmarco-v1-passage` (Lucene), k1=0.9, b=0.4 (Anserini defaults) for *all* query sets.
- **Cross-encoder** `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- **Bi-encoder** `sentence-transformers/msmarco-MiniLM-L6-cos-v5`.

## Deviations / implementation decisions

1. **Repo layout**: the repo root is `Graph_Project/` (the plan's `regat_fusion/` tree lives directly in it).
   Extra modules: `src/common.py` (config/IO), `src/store.py` (aligned tensor loading), `src/metrics.py`
   (fast torch NDCG/MRR used for early stopping / grid / oracle; `tests/test_metrics.py` checks it equals
   ir_measures). Final reported metrics use `ir_measures` (pytrec_eval backend).
2. **Cache format**: instead of `{qid: tensor[K]}` / `{pid: tensor[d]}` dicts (millions of tiny pickled tensors),
   caches are packed: scores `{'qids': [...], 'scores': Tensor[N, K]}` with rows in `cands_{set}.json` order;
   embeddings `{'ids': [...], 'emb': Tensor[N, d]}` (passages fp16 on disk, cast on load). Semantics unchanged.
3. **Candidate depth**: dev/small and TREC DL are retrieved to depth 1000 (only to report BM25 Recall@1000) and
   truncated to the top-100; train queries are retrieved to depth 100. Same BM25 run for everything.
   Runs with < 100 hits are padded with random passages scored below the last hit (count in
   `results/bm25_first_stage.json`, `n_padded_runs`).
4. **Protocol A sampling**: 75k judged train queries are sampled (seed 42); `val` = the first 2,000 (kept even
   without a positive in the top-100); `gat_train`/`fusion_train` are the next 30k/10k that have >= 1 positive
   in their top-100.
5. **Protocol B**: dev/small split 40/25/10/25 (seed 42); no-positive queries dropped from B's `gat_train` /
   `fusion_train` (listed in `splits_B.json:_dropped`). Protocol B trains its own GAT (`cache/B/gat_best.pt`);
   only `ctx_*` differs between protocols. TREC DL is evaluated under B as well (OOD), with GAT_B.
6. **Cross-encoder scoring** uses a plain HF `AutoModelForSequenceClassification` loop (raw logit, length-sorted
   batches, fp16) instead of `CrossEncoder.predict`; at start-up the first 64 logits are checked against
   `sentence_transformers.CrossEncoder.predict(activation_fn=Identity)` (must agree within 0.05).
7. **Bi-encoder** `max_seq_length = 256` (config); `s_bi` is computed from the stored fp16 embeddings so it
   matches exactly what the GAT sees.
8. **GAT**: graph self-loops are added explicitly, so `GATv2Conv(add_self_loops=False)` (otherwise PyG would
   duplicate them). On GPU, PyG scatter ops are not bit-deterministic; seeds are fixed but GAT training may
   differ in the last digits between reruns. The GAT is trained once per protocol and `train_gat` refuses to
   overwrite an existing checkpoint.
9. **random_graph control**: the symmetric kNN graph of each query with its node labels randomly permuted
   (A'[i, j] = A[pi(i), pi(j)]). The degree sequence and the number of edges are preserved exactly; which
   passages are neighbours becomes random w.r.t. content. (A first version drew `deg_i` random out-neighbours
   per node and symmetrised; that inflated the mean degree by ~25% because random edges are rarely mutual.)
   Optional `knn_bm25` control not implemented.
10. **alpha0** (best fixed alpha on val, ties -> smallest alpha) is clipped to [0.02, 0.98] before `logit()` for
    the MLP bias init / regulariser target, so the init is finite when the grid picks 0 or 1. Alpha is computed as
    `1e-6 + (1 - 2e-6) * sigmoid(z)` so it is strictly inside (0, 1) even when the sigmoid saturates.
11. **Temperature**: every config (incl. `fixed`) has its own learnable `log_T` (init T = 1).
12. **Paper row (`paper`)**: `fused = a*s_sem + b*s_ctx` (a, b free, init 0.5) + a learnable scalar bias c, loss
    = BCE(sigmoid(fused + c), y). The bias does not change the ranking but makes the pointwise BCE well-posed
    (without it, non-negative normalised scores such as minmax/rank can only lower negatives by shrinking a, b).
    Same for the BCE loss-ablation rows (`fixed`, `QS` with BCE): BCE(sigmoid(fused/T + c), y).
13. **Loss ablation** normaliser = the normaliser where `QS` (listwise) is best on val.
14. **"Best fixed"** baseline in `significance.csv` / `oracle_gap.csv` = the better (on val) of the grid-search
    fixed alpha and the learned scalar, under the same normaliser.
15. **Oracle**: the small-headroom flag (< 0.5 NDCG points) is decided on val (`results/{P}/oracle_summary.json`)
    and also reported on test (`oracle_gap.csv`).
16. **Listwise loss** is averaged over queries with >= 1 positive (all fusion_train queries have one).
17. **lambda = 0** runs (`lambda_extra`) are included for the three adaptive dependencies.
18. **TREC DL**: judged queries only; NDCG@10 on graded labels; MRR@10 and MAP with rel >= 2.
19. **Downloads**: the official MS MARCO host throttles each connection (~0.2 MB/s), so `src/download.py`
    pre-fetches the ir_datasets files with 32 parallel range requests (md5-verified against ir_datasets'
    own checksums) into `data/ir_datasets/`. The Pyserini index is served from Hugging Face.
20. **pyserini 1.6 + openai SDK**: pyserini builds an OpenAI client at import time with an empty key, which the
    current openai SDK rejects; `set_env` sets `OPENAI_API_KEY=unused-placeholder` (never used, BM25 only).
21. **Tie-breaking**: ties in fused scores (frequent with `rank` normalisation at round alphas) are broken by
    BM25 candidate order everywhere: the fast torch metric uses a stable sort, and runs are passed to
    ir_measures as rank-based scores so trec_eval's docid tie-breaking never kicks in.
22. **Significance** is run on test, TREC DL 2019 and 2020, for NDCG@10 and MRR@10.

## Results

Filled in automatically by `src/analysis.py` after the full runs.

<!-- RESULTS:A:BEGIN -->
(not run yet)
<!-- RESULTS:A:END -->

<!-- RESULTS:B:BEGIN -->
(not run yet)
<!-- RESULTS:B:END -->
