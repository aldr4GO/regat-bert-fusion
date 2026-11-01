# NOTES — deviations from the plan, VERIFY outcomes, results summary

Paper numbers (ReGAT-BERT, MS MARCO dev: MRR@10 0.413, NDCG@10 0.794) are **reported, protocol unknown**;
they are never mixed into our tables or significance tests (Sec. 3.0). The paper does not state split sizes,
candidate depth, negative sampling, graph construction on MS MARCO, GAT hyperparameters, or how Sim_sem is
bounded for the BCE; every comparison row is reproduced here under one protocol.

## VERIFY outcomes

All checked on 2026-10-31 with the pinned `requirements.txt` (torch 2.15, PyG 2.8.1, sentence-transformers 6.2,
pyserini 1.6, ir_datasets 0.6.3, ir_measures 0.4.3).

- **ir_datasets ids** resolve: `msmarco-passage/train/judged` (502,939 queries), `msmarco-passage/dev/small`
  (6,980), `msmarco-passage/trec-dl-2019/judged` (43), `msmarco-passage/trec-dl-2020/judged` (54), collection
  `msmarco-passage` (8,841,823 passages). Downloaded files match ir_datasets' md5 checksums.
- **BM25**: Pyserini prebuilt `msmarco-v1-passage` (Lucene, 8,841,823 docs) with k1=0.9, b=0.4 (Anserini
  defaults), used for *all* query sets. Smoke sample (200 dev/small queries): R@100 0.665, R@1000 0.895
  (published BM25 R@1000 on full dev/small: 0.857).
- **Cross-encoder** `cross-encoder/ms-marco-MiniLM-L-6-v2` resolves; our raw logits equal
  `CrossEncoder.predict(activation_fn=Identity)` to 2.3e-6.
- **Bi-encoder** `sentence-transformers/msmarco-MiniLM-L6-cos-v5` resolves (fallback not needed), dim 384.
- **Stage 1-2 acceptance** (smoke val, 60 queries): NDCG@10 CE 0.416 > s_bi 0.361 > BM25 0.223.
- **Stage 4 gates** (smoke): GAT-only val NDCG@10 0.373 vs s_bi 0.361 (A), 0.252 vs 0.244 (B); caches aligned.
- **Fast metrics** (`src/metrics.py`) match ir_measures exactly (unit test + BM25 val 0.22290 both ways).

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
22. **SLURM**: QOS `normal` allows 1 running / 3 submitted jobs per user (1 GPU, 32 CPU, 160 GB, 24 h), so
    the study runs as one resumable job (`slurm/run_pipeline.sbatch` -> `src/pipeline.py`) instead of a chain
    of per-stage jobs / a job array. Stages 0 and 5-6 therefore run inside a GPU allocation (CPU-only work);
    the GAT controls run last so the main results come first.
23. **Significance** is run on test, TREC DL 2019 and 2020, for NDCG@10 and MRR@10.

## Results

Filled in automatically by `src/analysis.py` after the full runs.

<!-- RESULTS:A:BEGIN -->
### Protocol A (auto-generated by `src/analysis.py`; selection on val, numbers on test)

- **Normalisation:** best (by val) is `minmax` (its best config `minmax_S_listwise_lam0.01`, test NDCG@10 0.4090). Per-normaliser test NDCG@10 of the val-best config: none 0.4091, minmax 0.4090, zscore 0.4085, rank 0.3940.
- **Alpha dependence:** test NDCG@10 averaged over normalisers: fixed 0.4054, Q 0.4042, QS 0.4039, S 0.4040 -> best `fixed`.
- **Adaptive vs best fixed (same normaliser, paired bootstrap, test NDCG@10):** 1/12 adaptive configs significantly better, 3 significantly worse. `none_Q_listwise_lam0.01` -0.0048 [-0.0062, -0.0034] p=0.0001; `none_QS_listwise_lam0.01` -0.0048 [-0.0062, -0.0034] p=0.0001; `none_S_listwise_lam0.01` -0.0049 [-0.0063, -0.0035] p=0.0001; `minmax_Q_listwise_lam0.01` +0.0000 [+0.0000, +0.0001] p=0.0402; `minmax_QS_listwise_lam0.01` +0.0000 [-0.0001, +0.0001] p=0.663; `minmax_S_listwise_lam0.01` +0.0000 [-0.0002, +0.0003] p=0.657; `zscore_Q_listwise_lam0.01` +0.0001 [-0.0000, +0.0001] p=0.0716; `zscore_QS_listwise_lam0.01` +0.0002 [+0.0001, +0.0003] p=0.001; `zscore_S_listwise_lam0.01` +0.0001 [-0.0001, +0.0003] p=0.16; `rank_Q_listwise_lam0.01` +0.0001 [-0.0001, +0.0003] p=0.398; `rank_QS_listwise_lam0.01` -0.0012 [-0.0027, +0.0002] p=0.0934; `rank_S_listwise_lam0.01` -0.0010 [-0.0023, +0.0002] p=0.0926
- **Oracle headroom (test, oracle minus best fixed, UPPER BOUND):** none +0.0474, minmax +0.0501, zscore +0.0506, rank +0.0647
- **CE over-confidence caveat:** mean top1-top2 logit gap train 0.740 vs dev 0.792.
<!-- RESULTS:A:END -->

<!-- RESULTS:B:BEGIN -->
### Protocol B (auto-generated by `src/analysis.py`; selection on val, numbers on test)

- **Normalisation:** best (by val) is `zscore` (its best config `zscore_QS_listwise_lam0.01`, test NDCG@10 0.4157). Per-normaliser test NDCG@10 of the val-best config: none 0.4170, minmax 0.4169, zscore 0.4157, rank 0.4150.
- **Alpha dependence:** test NDCG@10 averaged over normalisers: fixed 0.4163, Q 0.4164, QS 0.4162, S 0.4163 -> best `Q`.
- **Adaptive vs best fixed (same normaliser, paired bootstrap, test NDCG@10):** 0/12 adaptive configs significantly better, 0 significantly worse. `none_Q_listwise_lam0.01` +0.0001 [-0.0001, +0.0002] p=0.41; `none_QS_listwise_lam0.01` +0.0000 [-0.0000, +0.0001] p=0.521; `none_S_listwise_lam0.01` +0.0000 [-0.0002, +0.0002] p=0.844; `minmax_Q_listwise_lam0.01` +0.0000 [-0.0006, +0.0006] p=0.891; `minmax_QS_listwise_lam0.01` +0.0001 [-0.0001, +0.0004] p=0.288; `minmax_S_listwise_lam0.01` +0.0002 [+0.0000, +0.0004] p=0.0414; `zscore_Q_listwise_lam0.01` +0.0004 [+0.0000, +0.0009] p=0.0486; `zscore_QS_listwise_lam0.01` -0.0001 [-0.0022, +0.0020] p=0.94; `zscore_S_listwise_lam0.01` +0.0001 [-0.0007, +0.0010] p=0.761; `rank_Q_listwise_lam0.01` +0.0002 [-0.0005, +0.0010] p=0.581; `rank_QS_listwise_lam0.01` -0.0004 [-0.0016, +0.0007] p=0.534; `rank_S_listwise_lam0.01` -0.0002 [-0.0013, +0.0008] p=0.778
- **Oracle headroom (test, oracle minus best fixed, UPPER BOUND):** none +0.0398, minmax +0.0432, zscore +0.0442, rank +0.0463
- **CE over-confidence caveat:** mean top1-top2 logit gap train 0.740 vs dev 0.792.
<!-- RESULTS:B:END -->

---

# Multi-hop extension (MULTIHOP_EXTENSION_PLAN.md)

## Hypotheses (fixed before any multi-hop run, 2026-11-01)

- **H1 (alpha tracks complexity):** the learned alpha of `Q` / `QS` / `S` decreases as query complexity rises.
  Primary statistic per dataset, for the best adaptive variant (by val NDCG@10, primary normaliser):
  MuSiQue: Spearman(alpha, hop count); HotpotQA and BEIR HotpotQA: mean alpha(bridge) - mean alpha(comparison).
  Predicted < 0. Oracle alpha by hop count / type is reported next to it (independent of the gate).
- **H2 (adaptive gain is larger on multi-hop):** (best adaptive - best fixed NDCG@10 on test) minus the same gain on
  MS MARCO (protocol A); bootstrap over queries within each dataset. Predicted > 0.
- **H3 (more headroom):** (oracle - best fixed NDCG@10) minus the same on MS MARCO. Predicted > 0.
- **H4 (control):** H1-H3 statistics repeated with the GAT retrained on `none`, `random`, `title_link`, `hybrid`
  graphs (MuSiQue, HotpotQA). The effects should vanish for `none` / `random` if they come from context.
- H1-H3 on MuSiQue, HotpotQA and BEIR HotpotQA form the Holm family; 2WikiMultiHopQA, per-variant rows, H4 and
  everything else are exploratory and labelled so in `results/hypotheses.csv`.
- **Interpretation rule:** alpha is the relative reliability of the two signals under the chosen normaliser, not
  "how much context a query needs". Raw alpha is never compared across datasets without CE-only / GAT-only NDCG@10,
  the oracle alpha and alpha - alpha0 (all in `results/bucket_tables/`).

## Step 0: MS MARCO diagnosis (run 2026-11-01 on the existing caches)

`results/diagnostics_msmarco.{csv,json}`, figure `results/A/figures/diag_alpha_by_property.png`.
- CE-only beats GAT-only by only 0.016 val NDCG@10, so CE is not "much stronger" (threshold 0.05).
- Best fixed alpha moves a lot with the normaliser (none 0.05, minmax 0.70, zscore 0.70, rank 0.55): alpha is
  partly a scale artefact.
- The learned alpha barely adapts: test std 0.0003 (Q), 0.0012 (QS), 0.0063 (S) under minmax, far below 0.05.
- Spearman(alpha, n_words) and (alpha, n_entities) are slightly **positive** (+0.02 to +0.08, CIs above 0), i.e. the
  opposite of the predicted direction; the oracle alpha shows no reliable trend either.
- Graph controls: knn_emb beats random_graph by 0.0004 and no_graph by 0.001 val NDCG@10, so the MS MARCO
  contextual signal is not coming from graph structure.
- **Decision: the "simple queries" explanation is not supported on MS MARCO.** The multi-hop experiments still apply.
- **Primary normaliser = `minmax`** (wins on MS MARCO protocol A val; the plan expected zscore).
  Written to `results/primary_norm.json` and used for every primary multi-hop test.

## VERIFY outcomes (multi-hop)

- MuSiQue-Ans: HF `dgslibisey/MuSiQue` (`musique_ans_v1.0_{train,dev}.jsonl`, the official files): 19,938 train /
  2,417 dev; 20 paragraphs (a few pools have 18-19), field `is_supporting`; the hop count from the id prefix equals
  `len(question_decomposition)` for every question.
- HotpotQA distractor: HF `hotpotqa/hotpot_qa`, config `distractor` (parquet): 90,447 train / 7,405 validation;
  gold = title in `supporting_facts.title`; `type`, `level` present. Some dev pools have fewer than 10 paragraphs.
- 2WikiMultiHopQA: HF `framolfese/2WikiMultihopQA` (parquet): train + validation (12,576, used as test; the HF test
  split has no supporting facts); same context/supporting_facts layout as HotpotQA; `type` labels.
- BEIR HotpotQA: ir_datasets `beir/hotpotqa/{train,test}` + corpus (5,233,329 docs); BM25 = Pyserini prebuilt
  `beir-v1.0.0-hotpotqa.flat` (title + text), k1 = 0.9, b = 0.4, for train and test queries alike.
- spaCy `en_core_web_sm` (3.8) for the entity-count proxy.

## Deviations / decisions (multi-hop)

1. **Experiment keys** instead of new code paths: `A`/`B` (MS MARCO), `<dataset>`, `<dataset>@<variant>`
   (`none`, `random`, `title_link`, `hybrid`: H4; `msmarco_gat`: zero-shot E4). Only GAT-dependent files live
   under a variant (`cache/<ds>/graphs/<variant>/`, results in `results/<ds>@<variant>/`). The plan's `none`/`random`
   are the existing `no_graph`/`random_graph` builders.
2. **Masking:** every normaliser / feature / loss / metric / graph builder takes a candidate mask; with an all-valid
   mask the original code runs unchanged. Golden regression (subset, local CPU): all aggregate metrics and learned
   alphas identical (diff 0.0), per-query values within 3e-8 (float32 storage). The full grid is re-checked by the
   pipeline (`golden_A`, `golden_B`) and the new runs, which carry the extended metrics, replace the old ones
   (archived in `results/{A,B}/runs_v1/`).
3. **Metrics (user request):** nDCG, MRR, Precision and Recall @ {1, 3, 5, 10, 50} for every dataset (incl. MS MARCO),
   plus MAP, R@2 and AllRecall@{2,5,10} (plan Sec. 3.5). Precision@k divides by k (trec_eval), so P@50 on closed pools
   is just #gold/50 and R@50 = R@10 = 1 trivially for HotpotQA (K = 10); per-query arrays are stored only for
   ndcg@10, mrr@10, map, r@10, allr@5 (+ alpha) to keep run files small.
4. **Closed-pool candidate order and the "original order" baseline:** pools are sorted by in-pool BM25 (Pyserini
   default k1/b, IDF over all used paragraphs of the dataset, stop-worded + stemmed), which also serves as the BM25
   row and as the tie-breaker, exactly like the BM25 order on MS MARCO. The raw pool order of HotpotQA/MuSiQue is
   arbitrary, so it is not reported.
5. **Splits:** MuSiQue val = the rest of train after gat/fusion (2,938 questions); HotpotQA / 2Wiki val = 2,000 train
   questions; BEIR: 40,000 train queries retrieved, val = the first 2,000 (kept even without a positive), gat/fusion
   train need a positive in the top-100. Test = official dev (BEIR test).
6. **Graph:** k_nn = 3 for closed pools (DECISION), 8 for BEIR; same edge_min_sim 0.3. The k_nn sweep {2, 3, 5} is
   reported on val (`results/<ds>/gat_controls.csv`), the default is not changed after it. title_link scans each
   paragraph once with an alternation of the pool's normalised titles (leftmost-longest, non-overlapping matches),
   on the paragraph text only; computed for all closed-pool splits and for BEIR val/test (edge stats only).
   `mean_degree` now excludes the self-loop (the MS MARCO GAT summaries written earlier include it).
7. **Sanity gates:** enforced for the main key of every dataset (stop that dataset if GAT-only is more than 0.005
   val NDCG@10 below s_bi); reported only for graph-type / transfer variants. Dataset failures do not stop the other
   datasets in the pipeline.
8. **Zero-shot transfer** uses the frozen MS MARCO GAT with the target dataset's own graph (k_nn 3 on closed pools),
   the MS MARCO feature scaler and alpha0. Mixed / LODO gates use a scaler fitted on the union of fusion_train sets,
   early stopping on the union of val sets and alpha0 from a grid search on that union.
9. **H1 for HotpotQA / BEIR** compares bridge vs comparison (bridge predicted lower alpha); for 2Wiki the per-type
   alphas are in the bucket tables (exploratory). MuSiQue "type" = the id prefix (2hop, 3hop1, 3hop2, 4hop1, ...).
10. **E5 (`QSC`)** runs for every dataset (incl. MS MARCO) under the primary normaliser: QS + standardised
    [query length, spaCy entity count] computed from the query text only.

## Multi-hop results

<!-- HYPOTHESES:BEGIN -->
Auto-generated by `src/hypotheses.py` (primary normaliser `minmax`, test split, Holm over the primary H1-H3 tests).

- **H1 / musique** (spearman(alpha, hops), best=Q): +0.1067 [+0.0670, +0.1468], p=0.0001, Holm p=0.0009 -> **contradicted**
- **H2 / musique** (gain(ds) - gain(MS MARCO), best=Q): -0.0000 [-0.0002, +0.0002], p=0.694, Holm p=1 -> **inconclusive**
- **H3 / musique** (headroom(ds) - headroom(MS MARCO), oracle): +0.0244 [+0.0194, +0.0293], p=0.0001, Holm p=0.0009 -> **supported**
- **H1 / hotpotqa** (mean alpha bridge - comparison, best=QS): -0.0259 [-0.0365, -0.0155], p=0.0001, Holm p=0.0009 -> **supported**
- **H2 / hotpotqa** (gain(ds) - gain(MS MARCO), best=QS): +0.0001 [-0.0009, +0.0012], p=0.842, Holm p=1 -> **inconclusive**
- **H3 / hotpotqa** (headroom(ds) - headroom(MS MARCO), oracle): -0.0204 [-0.0237, -0.0171], p=0.0001, Holm p=0.0009 -> **contradicted**
- **H1 / beir_hotpotqa** (mean alpha bridge - comparison, best=Q): +0.0000 [+0.0000, +0.0000], p=0.0001, Holm p=0.0009 -> **contradicted**
- **H2 / beir_hotpotqa** (gain(ds) - gain(MS MARCO), best=Q): -0.0000 [-0.0002, +0.0002], p=0.652, Holm p=1 -> **inconclusive**
- **H3 / beir_hotpotqa** (headroom(ds) - headroom(MS MARCO), oracle): -0.0208 [-0.0243, -0.0174], p=0.0001, Holm p=0.0009 -> **contradicted**
- **H4 (exploratory controls):** see `results/hypotheses.csv` rows `H4 (...)`; the effects should vanish for `none` / `random` if they come from the graph.
<!-- HYPOTHESES:END -->

## Paper report (2026-11-01, `paper/`)

Built from the finished results only (no training, nothing selected on test):
`python -m src.report_analysis` (writes `results/report/*.csv`) then `python -m src.report_paper`
(writes `paper/tables/*.tex`, `paper/figures/*`, `paper/numbers.json`); `paper/main.tex` is the ACL-format text.

- **New analyses (exploratory, labelled so in the paper):** (a) fixed-alpha response curves for every normaliser on
  val and test; (b) per-bucket arg-max alpha with a 2,000-resample bootstrap CI and a tie-robust per-query preference
  (centre of the optimal-alpha set; queries whose NDCG@10 is flat in alpha are excluded); (c) "bucketed fixed alpha":
  one alpha per bucket chosen on val, applied on test, paired bootstrap vs alpha0. With gold hop/type buckets this uses
  labels that never enter a gate and are unavailable at test time, so it is reported as analysis, not as a method
  (plan Sec. 7 allows labels in analysis).
- **Gate collapse:** under min-max, 100% of the adaptive runs on MS MARCO A, MuSiQue and BEIR-HotpotQA restore their
  epoch-0 state (= the fixed-alpha initialisation) on val early stopping; this explains adaptive ~= fixed.
- **Val/test shift:** MS MARCO A and MuSiQue val are drawn from train questions (same distribution as GAT-train); the GAT is
  relatively stronger there (MuSiQue CE-GAT val -0.7 vs test +4.3 nDCG@10), so val alpha* (0.40) < test alpha* (0.65).
  Protocol B (val and test both from dev/small) shows no shift.
- **ReGAT-BERT reference:** the plans give no bibliographic reference; `paper/references.bib` has a marked placeholder
  (`regatbert`) to be completed by the authors.
- **LaTeX toolchain:** tectonic in a separate env `~/.conda/envs/tex` (not part of `requirements.txt`, which stays the
  experiment env); ACL style files from github.com/acl-org/acl-style-files (master).
