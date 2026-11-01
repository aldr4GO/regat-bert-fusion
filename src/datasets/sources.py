"""Downloads for the multi-hop datasets (HF dataset repos, BEIR via ir_datasets, Pyserini BEIR index)."""
from pathlib import Path

from src.common import ROOT, get_logger

log = get_logger("sources")


def raw_dir(ds):
    return ROOT / "data" / "raw" / ds


def hf_files(dcfg, split):
    """Local paths of the raw files of one split (downloaded on first use)."""
    from huggingface_hub import hf_hub_download
    return [Path(hf_hub_download(dcfg["hf_repo"], f, repo_type="dataset", local_dir=raw_dir(dcfg["name"])))
            for f in dcfg["files"][split]]


def download(cfg, ds):
    dcfg = {**cfg["multihop"]["datasets"][ds], "name": ds}
    if dcfg["kind"] == "closed":
        for split in dcfg["files"]:
            for p in hf_files(dcfg, split):
                log.info("%s/%s: %s (%.0f MB)", ds, split, p.name, p.stat().st_size / 1e6)
    else:
        import ir_datasets
        for key in ("train", "test"):
            d = ir_datasets.load(dcfg["ird"][key])
            n = sum(1 for _ in d.queries_iter())
            log.info("%s: %d queries, %d qrels", dcfg["ird"][key], n, sum(1 for _ in d.qrels_iter()))
        d = ir_datasets.load(dcfg["ird"]["corpus"])
        log.info("%s corpus: %d docs", ds, d.docs_count())
        from pyserini.search.lucene import LuceneSearcher
        s = LuceneSearcher.from_prebuilt_index(dcfg["pyserini_index"])
        log.info("%s pyserini index: %d docs", dcfg["pyserini_index"], s.num_docs)


def main():
    import argparse
    from src.common import load_config, set_env
    ap = argparse.ArgumentParser(description="download the raw files of a multi-hop dataset")
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()
    cfg = load_config()
    set_env(cfg)
    download(cfg, args.dataset)


if __name__ == "__main__":
    main()
