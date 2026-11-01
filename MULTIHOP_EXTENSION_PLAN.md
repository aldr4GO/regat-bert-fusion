# Extension Plan: Testing the "Multi-hop Needs More Context" Hypothesis

This extends `FUSION_ABLATION_PLAN.md`. Read that first; everything in it (frozen models, caching, normalizers, alpha variants, GAT, evaluation) stays unchanged unless this file says otherwise. **DECISION** = already made, implement it. **VERIFY** = check against the library or dataset, adapt, and log in `NOTES.md`.

Fusion convention (unchanged): `fused = alpha * s_sem_n + (1 - alpha) * s_ctx_n`, where `s_sem` is the frozen cross-encoder score and `s_ctx` is the cosine between the query embedding and the GAT-refined passage embedding.

---

## 0. Hypotheses (write these into NOTES.md before running anything)

- **H1 (alpha tracks complexity):** the learned alpha of the adaptive variants (`Q`, `QS`, `S`) decreases as query complexity rises (more hops, bridge vs comparison). Test: Spearman correlation between alpha and hop count, plus per-bucket mean alpha with bootstrap CIs.
- **H2 (adaptive gain is larger on multi-hop):** the gain of the best adaptive variant over the best fixed alpha is larger on multi-hop data than on MS MARCO. Test: paired bootstrap per dataset, then compare the gains.
- **H3 (more headroom):** the oracle-alpha gap (oracle minus best fixed) is larger on multi-hop data.
- **H4 (control):** if H1-H3 hold, they should vanish when the graph is replaced by a random graph or removed (`no_graph`). Otherwise the effect is not coming from context.

H1-H3 are the primary tests (correct with Holm). Everything else is exploratory and must be labelled so.

**Interpretation rule:** alpha measures the *relative reliability of the two signals under the chosen normalization*, not "how much context the query needs." Never compare raw alpha across datasets without also reporting (a) CE-only and GAT-only NDCG@10 per dataset and bucket, (b) the oracle alpha, (c) alpha minus alpha0 (the fixed-alpha optimum for that dataset).

---

## 1. Step 0: Diagnose the MS MARCO result first (no new data, about half a day)

Before adding datasets, check whether "alpha > 0.5 because MS MARCO queries are simple" is supported or whether something else explains it. Add `src/diagnose_msmarco.py` that reads the existing caches and writes `results/diagnostics_msmarco.csv` and plots:

1. **Signal strength:** NDCG@10 and MRR@10 of CE-only (alpha = 1) vs GAT-only (alpha = 0) on `val` and `test`. If CE is far stronger, alpha > 0.5 is expected for every query, with no link to simplicity.
2. **Normalization sensitivity:** best fixed alpha under each of `none`, `minmax`, `zscore`, `rank`. If it moves a lot, alpha is partly a scale artifact.
3. **Does alpha adapt at all?** Std of learned alpha across test queries, per variant and seed. Std below about 0.05 means the model barely adapts and cannot support complexity claims.
4. **Regularizer and init effect:** compare learned alpha with `lambda = 0.01` vs `lambda = 0`, and the mean learned alpha vs alpha0. If the mean sits on alpha0, the init and regularizer are dominating.
5. **Alpha vs query properties** (analysis only, never fed to the gate): query length (words), wh-word type (what/who/when/where/how/why/other), numeric-answer indicator (query contains "how many/much/long/old", "year"), number of named entities (spaCy `en_core_web_sm`). Report mean alpha and oracle alpha per bucket with bootstrap CIs.
6. **Graph control on MS MARCO:** re-run `no_graph` and `random_graph` (already in plan 6.6). If the contextual signal is no better than a random graph, MS MARCO alpha says nothing about context.

**Decision rule:** the "simple queries" explanation is supported only if alpha or oracle alpha varies with query properties in the direction the hypothesis predicts (items 3 and 5) *and* CE-only is not simply much stronger (item 1). Otherwise record "not supported on MS MARCO" and continue; the multi-hop experiments still apply.

---

## 2. Datasets

All three families give binary paragraph-level labels, so the loss, normalizers, and features carry over.

| Dataset | Setting | Candidates (K) | Complexity labels | Why |
|---|---|---|---|---|
| **MuSiQue-Ans** | Closed pool | 20 paragraphs per question (2-4 gold) | **Hop count (2/3/4)** from the question id prefix and `question_decomposition` length | Cleanest test of H1: gold hop count is annotated, and answers need composing 2-4 paragraphs |
| **HotpotQA (distractor)** | Closed pool | 10 paragraphs (2 gold) | `type` (bridge vs comparison), `level` (easy/medium/hard) | Large, standard, and bridge vs comparison is a natural complexity split |
| **BEIR HotpotQA** | Open retrieval (BM25 top-100 over about 5.2M abstracts) | 100 | Same labels joined by question id | Closest to the original MS MARCO pipeline; published BEIR baselines exist for sanity comparison (verify numbers) |
| **2WikiMultiHopQA** (optional) | Closed pool | about 10 paragraphs | `type` (comparison / bridge-comparison / compositional / inference) | Different question types, and a second source for the type test |
| **MS MARCO** (existing) | Open retrieval | 100 | Rule-based proxies only (Step 0) | The simple-query reference point |

**VERIFY** every dataset id and field name (`ir_datasets` has `beir/hotpotqa`; MuSiQue, 2Wiki, and HotpotQA distractor come from their official repos or Hugging Face `datasets`; split sizes below are approximate). Your sandbox may block downloads, so run data prep on your own machine.

**DECISION (priority order):** (1) MuSiQue-Ans and HotpotQA distractor, which are cheap and give the hypothesis test; (2) BEIR HotpotQA; (3) cross-dataset transfer; (4) 2Wiki only if time allows.

**Closed-pool caveat (state it in the write-up):** the gold paragraphs are always in the pool, so these settings test *reranking quality*, not retrieval. HotpotQA distractors are retrieved to resemble the question, so graph neighbors of a gold paragraph are often distractors; context may hurt. A negative result is a valid result.

### 2.1 Labels and splits (in-domain, official splits)

| Dataset | gat_train | fusion_train | val | test |
|---|---|---|---|---|
| MuSiQue-Ans | 10,000 from train | 7,000 from train | remaining train (about 2,900) | official dev (about 2,400) |
| HotpotQA distractor | 20,000 from train | 10,000 from train | 2,000 from train | official dev (7,405) |
| BEIR HotpotQA | 20,000 from train | 10,000 from train | 2,000 from train | BEIR test (7,405) |

Sampling seed 42, splits disjoint, saved to `cache/{dataset}/splits.json`. Relevance: a paragraph is relevant if it is a supporting paragraph (HotpotQA: its title appears in `supporting_facts`; MuSiQue: `is_supporting`). Keep queries with no relevant paragraph in the pool out of training, but in val and test.

---

## 3. Changes to the existing code

**Principle:** MS MARCO results must not change. Add a golden regression test (see 3.7).

### 3.1 Dataset adapters (`src/datasets/`)

Replace the MS MARCO-only `data.py` loader with a registry. Each adapter (`msmarco.py`, `musique.py`, `hotpotqa_distractor.py`, `beir_hotpotqa.py`, `twowiki.py`) returns the same object:

```
queries   {qid: text}
cands     {qid: [pid, ...]}                  # variable length
passages  {pid: {"title": str, "text": str}}  # title may be empty (MS MARCO)
labels    {qid: {pid: grade}}                # binary here
meta      {qid: {"hops": int|None, "type": str|None, "level": str|None, "n_words": int}}
splits    {"gat_train": [...], "fusion_train": [...], "val": [...], "test": [...]}
```

Pids must be unique per question in closed pools (prefix with qid, for example `f"{qid}::{i}"`), since the same paragraph can appear under many questions. Cache directory per dataset: `cache/{dataset}/`.

### 3.2 Variable candidate counts (the biggest change)

Closed pools have K = 10 or 20, not 100. Every module must support variable K:

- Cache tensors stored padded to `K_max` with a boolean `mask` `[K_max]`.
- `normalize.py`: min, max, mean, std, and rank computed over valid entries only; padded positions output 0 and are ignored later.
- `features.py`: all 12 features mask-aware. Top-m overlap uses `m = 10` when `K >= 50` and `m = 3` otherwise (config `overlap_m`). Spearman over valid entries only. Return finite values for `K < 4`.
- `fusion.py` / `train_fusion.py`: masked softmax in the listwise loss (set padded logits to `-inf`); the regularizer averaged per query.
- `evaluate.py`: ignore padded entries; if a query has fewer than 10 valid candidates, NDCG@10 and MRR@10 use all valid candidates.
- `oracle.py`, `analysis.py`: same masking.

### 3.3 Graph builder (`graph.py`)

- Parameter `k_nn` becomes per-dataset config: `k_nn = 8` for K = 100; **`k_nn = 3` for K <= 20** (with K = 10 and `k_nn = 8` the graph is nearly complete and the GAT sees everything, which makes "context" meaningless). Sweep `k_nn in {2, 3, 5}` on val for closed pools.
- Add edge builders selectable by `graph_type`:
  - `knn_emb` (default, existing)
  - `none` (self-loops only)
  - `random` (same degrees, random neighbors within the pool; seed fixed)
  - **`title_link`** (new, needs paragraph titles): edge between paragraphs i and j if the normalized title of j appears as a whole phrase in the text of i, or vice versa. Normalize by lowercasing and stripping parenthetical suffixes such as "(film)". Ignore titles shorter than 3 characters. This is a cheap proxy for the bridge-entity links multi-hop reasoning relies on. Expect noise from generic titles; log the average degree.
  - `hybrid` = union of `knn_emb` and `title_link` edges.
- Add `edge_stats.json` per dataset: mean degree, fraction of gold-gold pairs that are connected (for each graph type). This directly measures whether the graph links the supporting paragraphs. If `knn_emb` links gold pairs no more often than `random`, say so.

### 3.4 Models, caching, and GAT

- Cross-encoder, bi-encoder, and GAT architecture are unchanged and frozen exactly as before. **Same MiniLM checkpoints for all datasets**, so differences come from the data, not the model.
- **One GAT checkpoint per dataset**, each trained on that dataset's `gat_train` with the same recipe (InfoNCE, `tau = 0.05`, multi-positive). Closed pools have only 9-19 negatives per query, so listwise InfoNCE is weaker; keep it, and note it.
- **Transfer GAT:** also keep the MS MARCO GAT checkpoint and use it to compute `s_ctx` for multi-hop test queries (needed for the zero-shot experiment in 4.4).
- Include paragraph **title + text** as input to the cross-encoder and bi-encoder on HotpotQA/MuSiQue (`"{title}. {text}"`), since titles carry the entities. Config `use_title: true`; MS MARCO has none.
- Sanity gates before any fusion run (per dataset): CE-only beats BM25 or the pool's original order on `val`; the GAT-only score is not clearly worse than the raw `s_bi`; candidate order is identical across all caches.

### 3.5 Metrics

Keep NDCG@10, MRR@10, and MAP. Add, because multi-hop needs *all* gold paragraphs:

- **Recall@k** for k in {2, 5, 10} (open retrieval) and **joint Recall@k** (fraction of queries whose supporting paragraphs are *all* in the top k) for k in {2, 5, 10} (**AllRecall@k**).
- For closed pools with K = 10, Recall@10 is trivially 1; report R@2, AllR@2, AllR@5, NDCG@10, MAP and note this.
- Report first-stage BM25 Recall@100 and AllRecall@100 for BEIR HotpotQA (the pool's ceiling).

### 3.6 Analysis (`analysis.py`)

Add grouping by `meta`: hops (MuSiQue), type and level (HotpotQA, 2Wiki), n_words, and the Step 0 proxies. For each group report: mean learned alpha per variant, oracle alpha, CE-only NDCG, GAT-only NDCG, best fixed NDCG, adaptive NDCG, and the adaptive gain with a bootstrap CI. Plots: alpha vs hop count (box plots), oracle alpha vs hop count, and per-bucket gain bars, each across datasets.

### 3.7 Tests

- **Golden regression:** rerun the MS MARCO fusion grid on the existing cached tensors with the new masked code; all metrics and learned alphas must match the old results within 1e-6 (same seed). This is the acceptance gate for the refactor.
- **Padding invariance:** appending padded dummy candidates to a query changes none of its normalized scores, features, loss, or metrics.
- Variable-K edge cases: K = 1, K = 2, queries whose scores are all equal, and queries with zero relevant paragraphs.
- Graph tests: `title_link` is symmetric after symmetrization, has no cross-query edges, and `random` preserves the degree sequence.
- Label-perturbation leakage test (unchanged).

---

## 4. Experiments

**DECISION (limit forking paths):** the primary normalization is the one that wins on MS MARCO `val` (expected `zscore`); fix it now and use it for all primary multi-hop tests. Other normalizations are reported as secondary. Hyperparameters are selected on each dataset's `val` only; `test` is touched once at the end.

### 4.1 E1: In-domain grid on closed-pool datasets (MuSiQue, HotpotQA distractor)

Per dataset: GAT trained in-domain, then the fusion grid `{fixed, Q, QS, S}` x primary normalization (full 4 x 4 grid as secondary), 5 seeds, `paper` baseline, CE-only, GAT-only, original pool order or BM25, and oracle alpha. Run the oracle analysis first, because it shows the headroom (H3) before you spend time elsewhere.

### 4.2 E2: Open retrieval (BEIR HotpotQA)

Same as E1, with BM25 top-100 candidates, `k_nn = 8`, and the same BM25 implementation for train and test queries. Report AllRecall@k and compare with published BEIR numbers only as a sanity check (different protocol).

### 4.3 E3: Hypothesis tests (H1, H2, H3, H4)

- **H1:** Spearman rho between learned alpha and hop count (MuSiQue), and alpha by bridge vs comparison (HotpotQA), for `Q`, `QS`, `S`; mean over seeds, bootstrap CI over queries. Report oracle alpha by hop count too, since it does not depend on the gate.
- **H2:** paired bootstrap of adaptive minus best-fixed NDCG@10 per dataset; then the difference of these gains between each multi-hop dataset and MS MARCO (bootstrap over queries within each dataset).
- **H3:** oracle minus best-fixed per dataset with CIs.
- **H4:** repeat E1 with `graph_type` in `{none, random, knn_emb, title_link, hybrid}` (retrain the GAT for each; 5 graph types x 2 closed-pool datasets, about 20 min each).
- Holm-correct H1-H3 across datasets; label everything else exploratory.

### 4.4 E4: Transfer and mixed training

1. **Zero-shot:** train the gate (all variants) on MS MARCO `fusion_train`, apply to multi-hop `test` with the MS MARCO GAT's `s_ctx`. Does the gate, trained on simple queries, shift alpha on multi-hop queries without seeing them? Compare with the in-domain gate.
2. **Mixed:** train one gate on the union of `fusion_train` sets (MS MARCO + MuSiQue + HotpotQA), add no dataset-id feature, test on each. A single gate that matches in-domain gates supports the "alpha adapts to the query" story.
3. **Leave-one-dataset-out:** train on MS MARCO + HotpotQA, test on MuSiQue (3-4 hop questions never seen).

### 4.5 E5 (optional): explicit complexity features

Add one more gate variant `QS+C`: `QS` plus query length and entity count. If `QS+C` does not beat `QS`, the learned gate already captures complexity from the embedding. Never include these features in the main `Q/QS/S` variants.

### 4.6 E6 (optional): 2WikiMultiHopQA

Same as E1 on closed pools, with the `type` labels.

---

## 5. Outputs

- `results/{dataset}/main_table.csv`: all configs plus baselines, metrics from 3.5.
- `results/hypotheses.csv`: for H1-H4, the statistic, CI, raw p, and Holm-adjusted p.
- `results/bucket_tables/{dataset}_{label}.csv`: alpha, oracle alpha, CE-only, GAT-only, fixed, adaptive, and gain per bucket.
- `results/edge_stats.csv`: gold-gold connectivity per graph type and dataset.
- `results/transfer.csv`: E4.
- Figures: alpha vs hops (box plots), oracle alpha vs hops, bucket gains, graph-type comparison.
- `NOTES.md`: for each hypothesis, supported / not supported / inconclusive, with the numbers, plus every deviation.

---

## 6. Order of work and acceptance criteria

1. Step 0 diagnostics on MS MARCO. **Accept:** `diagnostics_msmarco.csv` written; decision on the "simple queries" explanation recorded.
2. Refactor for adapters and masking. **Accept:** golden regression and padding tests pass.
3. MuSiQue and HotpotQA distractor: data prep, caches, sanity gates. **Accept:** gates pass; `edge_stats.json` computed.
4. E1 and the oracle analysis on those two. **Accept:** H3 numbers available; stop and reassess if oracle headroom is under 0.5 NDCG points everywhere.
5. E3 (H1, H2, H4 graph controls).
6. BEIR HotpotQA (E2), then E4 transfer.
7. Optional E5/E6, then the write-up.

**Rough effort (estimate):** closed-pool caches take minutes of GPU time (about 0.4M pairs per dataset); BEIR HotpotQA needs a BM25 index over about 5.2M abstracts and about 2.7M cross-encoder pairs (about 1-2 h on the RTX 6000 Ada). Total about 1-1.5 weeks including the refactor and debugging.

---

## 7. Rules (in addition to those in the base plan)

- No tuning on any `test` split. No dataset-specific cross-encoder or bi-encoder swaps in the main results.
- Do not fit or select anything on hop-count labels except in the analysis; labels never enter the gate inputs (except the optional `QS+C` features, which are computed from the query text, not from gold labels).
- If the multi-hop gains are small, report them as small. A clear null result with the oracle headroom and graph controls is still a publishable finding.
