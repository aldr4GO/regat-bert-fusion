# Implementation Plan: Score-Fusion Ablations for ReGAT-BERT

Paste this file into the repo root (or reference it from CLAUDE.md). Follow it literally. Where a choice is marked **DECISION**, it is already made; do not ask, just implement it. Where marked **VERIFY**, check it against the installed library or dataset and adapt, then log what you changed in `NOTES.md`.

---

## 0. Goal

ReGAT-BERT fuses two scores with one global weight:

    final(q, p) = alpha * s_sem(q, p) + (1 - alpha) * s_ctx(q, p)

- `s_sem`: cross-encoder BERT relevance score (query-passage joint encoding).
- `s_ctx`: cosine similarity between the query embedding and the **GAT-refined** passage embedding.

This study changes **only the fusion step**. It compares two axes:

1. **Normalization** of the two scores into a common space (4 options).
2. **What alpha depends on** (4 options): global constant, query only, query + both scores, both scores only.

Everything upstream of fusion is trained or computed once, then **frozen and cached**. Fusion experiments run on cached tensors, so each run takes seconds to minutes on CPU.

Out of scope: changing the cross-encoder, the bi-encoder, the GAT architecture, or the retrieval stage between ablations. All ablations must share exactly the same cached scores.

---

## 1. Pipeline overview

```
Stage 0  Data + candidates          (BM25 top-100 per query, from ir_datasets)
Stage 1  Cache semantic scores      (frozen cross-encoder -> s_sem)
Stage 2  Cache embeddings           (frozen bi-encoder -> q_emb, p_emb)
Stage 3  Train the GAT once         (encoders frozen; InfoNCE) -> freeze it
Stage 4  Cache contextual scores    (frozen GAT -> s_ctx)
Stage 5  Fusion ablations           (only the fusion module is trainable)
Stage 6  Evaluation + analysis
```

### What is frozen, and when

| Component | Status | Notes |
|---|---|---|
| Cross-encoder | **Frozen always** (never trained here) | Used as an off-the-shelf pretrained model |
| Bi-encoder | **Frozen always** | Produces `q_emb`, `p_emb` |
| GAT | **Trained once in Stage 3, then frozen** for every Stage 5 run | Same checkpoint for all 16 configs |
| Fusion module (alpha network, learnable temperature) | **Only trainable part in Stage 5** | |

Why GAT is trained once and then frozen: if the GAT changed between configs, differences could come from the GAT and not from the fusion strategy.

---

## 2. Environment and repo structure

**Python 3.10+**. Packages: `torch`, `torch_geometric` (PyG), `sentence-transformers`, `ir_datasets`, `pytrec_eval` (or `ir_measures`), `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `pyyaml`, `tqdm`, `pytest`.

**DECISION (compute):** Stages 1-4 need a GPU (Colab/Kaggle is fine); Stage 5-6 run on CPU. Stages 1-4 must write their outputs to disk so that Stage 5 never needs the GPU or the models.

```
regat_fusion/
  CLAUDE.md                  # short pointer to this plan
  NOTES.md                   # log of deviations, VERIFY outcomes
  configs/
    data.yaml  gat.yaml  fusion.yaml
  src/
    data.py                  # load queries, qrels, candidates, splits
    cache_semantic.py        # Stage 1
    cache_embeddings.py      # Stage 2
    graph.py                 # kNN candidate graph builder
    gat.py                   # GAT module
    train_gat.py             # Stage 3
    cache_contextual.py      # Stage 4
    normalize.py             # 4 normalizers
    features.py              # score-statistic features
    fusion.py                # alpha networks + fusion model
    train_fusion.py          # Stage 5
    oracle.py                # per-query oracle alpha
    evaluate.py              # metrics + significance
    analysis.py              # alpha plots, buckets
    run_all.py               # runs the 16-config grid x seeds
  tests/
  cache/                     # .pt / .npz files (gitignored)
  results/                   # csv, json, figures
```

---

## 3. Stage 0: Data, ground truth, candidates, splits

### 3.0 What the paper states (ReGAT-BERT, Sec. 3.4 and 4.1) and what it leaves out

Stated: MS MARCO Passage Ranking (8.8M passages, about 1M training queries, plus a dev set); metrics MRR@10 and NDCG@10 (Recall@1000 in Table 2); ground truth is the **binary** relevance label y_i (1 relevant, 0 not) used in a pointwise BCE loss (Eq. 6); fusion is `alpha * Sim_sem + beta * Sim_ctx` with both weights free and learnable (Eq. 5).

**Not stated** (cannot be reproduced exactly): train/validation/test sizes, whether a validation set exists, whether "dev" means dev.small or full dev, candidate depth K, negative sampling, how the graph is built on MS MARCO, GAT depth/heads/hyperparameters, how Sim_sem is bounded for the BCE.

**Consequence:** use the same dataset, the same ground truth and the same metrics as the paper, but **reproduce every comparison row ourselves under one protocol**. Paper numbers are listed as "reported, protocol unknown" and are never mixed into significance tests. Log all of this in `NOTES.md`.

### 3.1 Protocol A (primary, paper-style: train on train, test on dev)

- Dataset: `msmarco-passage` via `ir_datasets`. Ground truth: the **official binary qrels**, same as the paper. **VERIFY** ids.
- **Candidates for all queries (train and dev) come from the same BM25 implementation** (Pyserini prebuilt `msmarco-v1-passage`, or `bm25s` as fallback), top K = 100. Do not use the official top-1000 file for dev and a different BM25 for train, because that creates a train/test candidate mismatch.
- Splits:

| Split | Source | Size | Used for |
|---|---|---|---|
| `gat_train` | MS MARCO **train** queries | 30,000 | Train the GAT (Stage 3) |
| `fusion_train` | MS MARCO **train** queries, disjoint from `gat_train` | 10,000 | Train fusion modules (Stage 5) |
| `val` | MS MARCO **train** queries, disjoint from both | 2,000 | Early stopping, fixed-alpha grid, hyperparameters |
| `test` | official `dev/small` | 6,980 | Final reporting only (the paper's evaluation set) |

- Drop queries with no relevant passage inside their top-100 from `gat_train` and `fusion_train`; **keep** them in `val` and `test` (they score 0).
- **Known caveat (state it in the write-up):** the public MiniLM cross-encoder was trained on MS MARCO train, so `s_sem` is over-confident on train queries relative to dev. Log the score-distribution difference (mean top-1 minus top-2 gap, train vs dev). The paper has the same issue if its cross-encoder was fine-tuned on train; it does not say.

### 3.2 Protocol B (secondary, leak-free robustness check)

Split `dev/small` by seed 42 into `gat_train` 40% / `fusion_train` 25% / `val` 10% / `test` 25%. Nothing here was seen by the cross-encoder. Report Protocol B next to A; if the conclusions agree, the contamination caveat is not driving the result. TREC DL 2019/2020 may be added as out-of-distribution tests (not in the paper; label them so).

### 3.3 Reproduced comparison rows (all under the same protocol)

BM25; SBERT (`s_bi`); CE-BERT (alpha = 1); GAT-only (alpha = 0); **ReGAT-BERT paper-style** (see 7.2, row `paper`); then our variants. Candidate depth K = 100 means Recall@1000 is not reproducible for rerankers; report first-stage BM25 Recall@100 and @1000 only.

**Sanity flag for the write-up:** the paper reports NDCG@10 = 0.794 with MRR@10 = 0.413. On MS MARCO dev with sparse binary labels, NDCG@10 normally sits only a few points above MRR@10, and Table 2 lists identical Recall@1000 for BM25 and ColBERT (0.814). The baseline rows therefore look drawn from different protocols, which is another reason to reproduce them.

---

## 4. Stage 1: Frozen cross-encoder scores (`s_sem`)

- **DECISION (model):** `cross-encoder/ms-marco-MiniLM-L-6-v2` via `sentence_transformers.CrossEncoder`. Put the model name in config so it can be swapped for a BERT-base cross-encoder later. **VERIFY** the model id resolves.
- Score every (query, passage) pair for all splits and TREC DL sets. Use fp16 on GPU, `max_length=256`, batch size 128, `model.eval()`, `torch.no_grad()`.
- Save the **raw logit** (no sigmoid) as `cache/sem_{split}.pt`: a dict `{qid: tensor[K]}` aligned with a fixed candidate order `cache/cands_{split}.json` (`{qid: [pid, ...]}`).

---

## 5. Stage 2: Frozen bi-encoder embeddings

- **DECISION (model):** `sentence-transformers/msmarco-MiniLM-L6-cos-v5` (**VERIFY** id; fall back to `multi-qa-MiniLM-L6-cos-v1`). Embeddings are L2-normalized.
- Encode every unique query and every unique candidate passage once. Save `cache/q_emb.pt` (`{qid: tensor[d]}`) and `cache/p_emb.pt` (`{pid: tensor[d]}`), float32 (store fp16 on disk if needed, cast back on load).
- Also compute and save the raw bi-encoder cosine `s_bi(q, p)` per candidate (used only as an extra baseline, not in fusion).

---

## 6. Graph construction and GAT (Stages 3-4)

### 6.1 Why this graph

MS MARCO passages have no section hierarchy, so the paper's structural graph cannot be reproduced. **DECISION:** use a **per-query candidate graph**: for each query, the graph is built only over its K = 100 candidates. It captures redundancy and topical coherence among candidates, and is cheap.

### 6.2 Graph definition (implement in `graph.py`)

- **Nodes:** the K candidate passages of one query. Node feature = frozen bi-encoder embedding `p_emb` (dim d, e.g. 384).
- **Edges:** for each node, connect to its **k_nn = 8** nearest neighbors by cosine similarity **among the other candidates of the same query**. Make the graph **undirected** (symmetrize), add **self-loops**. Drop edges with cosine < 0.3 (config `edge_min_sim`), but never drop self-loops.
- **The query is NOT a node** and does not influence the graph. This guarantees the graph depends only on the candidate set, and `s_ctx` differs from `s_sem` only through the structural signal and the final cosine with the query.
- Edge attributes: none (plain GAT). Build with torch tensors (`torch.cdist`/matmul on the normalized embeddings, `topk`), not sklearn, for speed. Output a PyG `Data` object per query, or a batched equivalent.

### 6.3 GAT module (`gat.py`)

```
input  h0 = p_emb                          [K, d]
layer1 = GATv2Conv(d, 128, heads=4, concat=True, dropout=0.1)   -> [K, 512]
act    = ELU
layer2 = GATv2Conv(512, d, heads=1, concat=False, dropout=0.1)  -> [K, d]
h      = LayerNorm(h0 + layer2_output)     # residual connection
h_ref  = L2-normalize(h)
s_ctx(q, p) = cos(q_emb, h_ref[p])
```

The residual connection matters: with it, an untrained GAT starts near the plain bi-encoder score, so `s_ctx` is a sensible signal from step 0. Also expose the attention weights (`return_attention_weights=True`) for later analysis.

### 6.4 GAT training (`train_gat.py`, `gat_train` split only)

- Loss: **listwise InfoNCE** over the K candidates of each query:
  `L = -log( sum_{p in positives} exp(s_ctx(q,p)/tau) / sum_{p in all K} exp(s_ctx(q,p)/tau) )`, with `tau = 0.05`.
- This uses all other candidates (which are BM25 hard negatives) as negatives. Accept the false-negative risk of sparse MS MARCO labels, and note it in the write-up.
- AdamW, lr 1e-4, weight decay 0.01, batch of 32 queries, up to 20 epochs, early stopping (patience 3) on `val` NDCG@10 of `s_ctx` alone.
- After training: `model.eval()`, `requires_grad_(False)`, save to `cache/gat_best.pt`. **Never touch it again.**

### 6.5 Stage 4: cache contextual scores

Run the frozen GAT over every split (including `gat_train`) and TREC DL. Save `cache/ctx_{split}.pt` (`{qid: tensor[K]}`), aligned with the same candidate order.

**Sanity gates (must pass before Stage 5):**
- GAT-only NDCG@10 on `val` should be at least as good as the raw bi-encoder score `s_bi`. If it is clearly worse, stop and debug (check the residual connection, tau, the edge threshold) before continuing.
- Candidate order and length are identical across `sem_`, `ctx_`, and `cands_` files (assert).

### 6.6 Why this graph, and the controls that test it

**Justification (details in the write-up):** MS MARCO passages have no section hierarchy or citation links, so the paper's edge types ("intra-document hierarchies, cross-references") do not exist here and the paper gives no construction for this dataset. We therefore use the content-based analogue: a per-query kNN graph over the candidate passages, using the same SBERT embeddings the paper uses as node features. It is label-free, cheap (100 nodes per query), consistent with the paper's relevance-propagation filter (context only matters among surviving candidates), and rooted in the cluster hypothesis and score-regularization work (Kurland and Lee 2005; Diaz 2005; MacAvaney et al. 2022 corpus-graph reranking). It is an **adaptation, not a replication**; say so.

**Controls (run on `val`, report in the paper's ablation table; all use the same GAT training recipe):**
1. `no_graph`: self-loops only (GAT reduces to a per-node transform).
2. `random_graph`: same node degrees as the kNN graph, random neighbors within the candidate set.
3. `knn_emb` (default): k_nn = 8, min cosine 0.3.
4. Sensitivity: k_nn in {4, 8, 16} and min cosine in {0.0, 0.3, 0.5}.
5. Optional: `knn_bm25` (edges from lexical passage-passage similarity).

If `random_graph` is within noise of `knn_emb`, the contextual gain is not coming from structure; report that honestly. The defaults (k_nn = 8, 0.3) are design choices, not paper values; the sensitivity sweep is what justifies them.

---

## 7. Stage 5: The fusion ablation

All of this operates on the cached tensors only. Per-query inputs are:

```
q_emb      [d]
s_sem_raw  [K]     # cross-encoder logits
s_ctx_raw  [K]     # GAT cosine scores
labels     [K]     # 0/1 (training); graded (TREC DL eval)
```

### 7.1 Axis 1: Normalization (`normalize.py`)

All normalizers are applied **per query, across that query's K candidates**, separately to each score, producing `s_sem_n` and `s_ctx_n`. Each takes `[K]` and returns `[K]`. Use `eps = 1e-8`.

| Name | Formula |
|---|---|
| `none` | identity |
| `minmax` | `(x - min) / (max - min + eps)` |
| `zscore` | `(x - mean) / (std + eps)` |
| `rank` | `1 - (rank(x) - 1) / (K - 1)` where rank 1 = highest score; ties get the average rank |

Unit tests: `minmax` output in [0, 1]; `zscore` has mean about 0 and std about 1; `rank` is invariant to any monotone transform of the input; all four preserve the argsort order of each individual score.

### 7.2 Axis 2: What alpha depends on (`fusion.py`)

The fused score is always

    fused = alpha * s_sem_n + (1 - alpha) * s_ctx_n        # alpha in (0, 1), one scalar per query

| ID | Name | Alpha |
|---|---|---|
| `fixed` | Global constant | A single learned or grid-searched scalar, same for all queries (see below) |
| `Q` | Query only | `alpha = sigmoid( MLP_Q( q_emb ) )` |
| `QS` | Query + both scores | `alpha = sigmoid( MLP_QS( concat( q_emb, feats(s_sem_raw, s_ctx_raw) ) ) )` |
| `S` | Both scores only | `alpha = sigmoid( MLP_S( feats(s_sem_raw, s_ctx_raw) ) )` |

**MLP architecture (all three):** `Linear(in, 64) -> ReLU -> Dropout(0.1) -> Linear(64, 1)`, then sigmoid. **Initialize** the last layer weights to near zero (std 0.01) and its **bias to `logit(alpha0)`** where `alpha0` is the best fixed alpha found on `val` for the same normalization. This makes every adaptive model start as the fixed baseline, so any gain is attributable to the learned dependence.

**`fixed`:** do two things and report both: (a) **grid search** alpha in {0.0, 0.05, ..., 1.0} on `val`, choosing max NDCG@10 (this is the paper-style fixed alpha and the strongest simple baseline); (b) the learned-scalar version using the same training loop (a single sigmoid(parameter)). Use (a) as the main baseline.


### 7.2b Paper-faithful baseline (`paper`) and loss ablation

Row `paper`: `fused = a * s_sem + b * s_ctx`, with `a` and `b` free scalars (init 0.5 each, not constrained to sum to 1), trained with the paper's pointwise **BCE** on binary labels (Eq. 6). BCE needs a probability, which the paper does not define, so apply `sigmoid(fused)`; log this assumption. Run `paper` on raw scores and on each normalizer.

Loss ablation (small, 2 rows per loss): run `fixed` and the best adaptive config (`QS`) with both (i) the listwise loss in 7.4 and (ii) the paper's BCE. This separates "adaptive alpha helps" from "the listwise loss helps".

### 7.3 Score-statistic features (`features.py`)

**DECISION:** features are computed from the **raw** scores (`s_sem_raw`, `s_ctx_raw`), **not** the normalized ones. This keeps the feature set identical across the normalization ablation, so the normalization axis stays clean. For each query compute, with `T` the top-K list and softmax taken over the K candidates after per-query z-scoring:

For **each** of the two scores (`sem`, `ctx`), 5 features:
1. mean of raw scores
2. std of raw scores
3. max of raw scores
4. gap between rank-1 and rank-2 raw scores
5. entropy of the softmax distribution over K (after z-scoring)

Plus 2 cross-score features:
11. Spearman correlation between `s_sem_raw` and `s_ctx_raw`
12. overlap of the top-10 sets of the two scores, divided by 10

Total: **12 features**. Standardize every feature to zero mean and unit variance using statistics from `fusion_train` only (save them, apply to val/test). Query-length features are **not** included (keep `Q` purely embedding-based so the three adaptive variants differ only in their inputs).

### 7.4 Fusion training (`train_fusion.py`)

- Trainable: the alpha MLP and one **learnable log-temperature** `log_T` (shared across all configs, including `fixed`, for fairness; initialize T = 1).
- Final scores for the loss: `fused / T`.
- **Loss: listwise multi-positive cross-entropy** over the K candidates:
  `L = -log( sum_{p in pos} softmax(fused/T)_p )` averaged over queries.
- **Regularizer:** `lambda * (alpha - alpha0)^2` averaged over queries, `lambda = 0.01` (config; if time allows, also run `lambda = 0`). It prevents collapse to 0 or 1.
- Optimizer AdamW, lr 1e-3, weight decay 1e-2, batch 64 queries, max 50 epochs, early stopping (patience 5) on `val` NDCG@10 of the **fused ranking**. Restore the best epoch.
- **Seeds:** 5 (0-4). Deterministic: set `torch`, `numpy`, `random` seeds, and `torch.use_deterministic_algorithms(True)` where possible.
- **Label leakage guard:** features and inputs must never use `labels`. Add a test that perturbs labels and asserts that alpha is unchanged.

### 7.5 The experiment grid

Normalization (4) x alpha dependence (4) = **16 configs**. Adaptive configs (`Q`, `QS`, `S`) run 5 seeds each; `fixed` is run with 5 seeds for the learned-scalar variant and once for the grid-search variant.

Reference rows (no training), evaluated on `test`:
- BM25 (first-stage ranking)
- `s_bi` alone (bi-encoder cosine)
- Cross-encoder alone (`alpha = 1`)
- GAT-only (`alpha = 0`)
- **Oracle alpha**: for each query, choose alpha from {0, 0.05, ..., 1} maximizing that query's NDCG@10 (ties: choose the alpha closest to `alpha0`). Computed per normalization. It is an **upper bound, not a method**; label it as such. Run `oracle.py` first: if oracle minus best-fixed is under 0.5 NDCG points, report this prominently, because it caps what adaptive alpha can gain.

---

## 8. Stage 6: Evaluation and analysis

**Metrics (`evaluate.py`, via `pytrec_eval` or `ir_measures`):** MRR@10, NDCG@10, MAP, and Recall@100 (which is fixed by the candidate set; report it once). On TREC DL use graded qrels for NDCG@10 and binary (relevance >= 2) for MRR@10 and MAP. Evaluate on `test` and on both TREC DL sets.

**Significance:** for each adaptive config vs the best fixed config **under the same normalization**: per-query paired bootstrap (10,000 resamples) over the `test` queries, using the per-seed-averaged per-query score. Report the mean difference, a 95% CI, and a p-value. Also report mean +/- std over seeds.

**Required outputs (`results/`):**
1. `main_table.csv`: rows = 16 configs plus reference rows; columns = MRR@10, NDCG@10, MAP (mean and std over seeds) on `test`, TREC DL 2019, TREC DL 2020.
2. `significance.csv`: paired tests described above.
3. `oracle_gap.csv`: oracle vs best fixed vs each adaptive method.
4. Plots (matplotlib, saved as PNG and PDF):
   - heatmap of NDCG@10 over normalization x alpha dependence
   - histogram of learned alpha per adaptive method
   - alpha vs query length (words) scatter with a trend line
   - alpha vs cross-encoder confidence (top1-top2 gap) scatter
   - NDCG@10 by query-length bucket (short <= 4 words, medium 5-7, long >= 8) for fixed vs `Q` vs `QS` vs `S`
5. `NOTES.md` summary: which normalization wins, which alpha dependence wins, whether adaptive beats fixed beyond noise, and the oracle headroom.

---

## 9. Tests (`tests/`) and sanity checks

- `alpha = 1` reproduces the cross-encoder ranking exactly; `alpha = 0` reproduces the GAT ranking exactly (for every normalizer, since all preserve order).
- The graph is symmetric, has self-loops, has no cross-query edges, and every node has degree >= 1.
- Alpha is always strictly in (0, 1).
- Feature vector has length 12 with no NaN/inf, even for a query where all scores in a list are identical (std = 0; the `eps` handling must hold).
- Rerunning the same config with the same seed gives identical results.
- Metric sanity: a perfect ranking gives NDCG@10 = 1 and MRR@10 = 1; a hand-built toy example matches a manual computation.
- Split files: the four splits are disjoint and cover all queries.
- Label-perturbation test (see 7.4).

---

## 10. Order of work and acceptance criteria

1. Scaffold the repo, configs, and tests; build splits (Stage 0). **Accept:** splits file written, candidate recall@100 reported.
2. Stages 1-2 caches. **Accept:** CE alone beats BM25 and `s_bi` on `val`. If not, stop and check the model id and `max_length`.
3. Graph + GAT (Stage 3-4). **Accept:** sanity gates in 6.5.
4. Normalizers, features, fusion module + unit tests. **Accept:** all tests in section 9 pass.
5. Oracle analysis and fixed-alpha grid on `val`.
6. Full 16-config grid x seeds (`run_all.py`), logging each run to `results/runs/*.json`.
7. Evaluation, significance, plots, `NOTES.md`.

---

## 11. Rules for the implementer

- Use python environment to install the libraries and maintain a requirements.txt file.
- Utilize the GPU available in the device (use nvidia-smi to check) for training/parallel computing, etc. I want the implementation to be as fast as possible, given that it should be correct.
- Implement the plan in phases so that I can keep a check of the module/phase that has been completed. It would be great if you could manage the version control history using git. Commiting after every phase is completed.
- Never train or fine-tune the cross-encoder or bi-encoder. Never retrain the GAT after Stage 3.
- Never use `test` or TREC DL for any decision (early stopping, alpha selection, hyperparameters). Use `val` only.
- Never change the candidate order between files.
- Keep every hyperparameter in `configs/*.yaml`; no magic numbers in code.
- Log every deviation from this plan (and every **VERIFY** outcome) in `NOTES.md` before continuing.
- If something in this plan is infeasible (for example, a dataset id is unavailable), choose the closest simple alternative, log it, and continue; don't stop to ask unless it changes the experimental design.
