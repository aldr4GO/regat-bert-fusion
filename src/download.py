"""Pre-fetch the large MS MARCO files with parallel HTTP range requests (the official host throttles each
connection to ~0.2 MB/s), verify their md5 and place them where ir_datasets expects them. Also triggers the
Pyserini prebuilt-index download. Stage 0 works without this step, just much slower.

Usage:  python -m src.download [--connections 32]
"""
import argparse
import hashlib
import json
import os
import shutil
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.common import ROOT, get_logger, load_config, set_env

log = get_logger("download")
IRDS_KEYS = ["collectionandqueries", "queries", "train/qrels"]   # msmarco-passage files over ~10 MB


def _size(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])


def _fetch_range(url, start, end, part, retries=20):
    for attempt in range(retries):
        have = part.stat().st_size if part.exists() else 0
        if start + have > end:
            return
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start + have}-{end}"})
            with urllib.request.urlopen(req, timeout=120) as r, open(part, "ab") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            if part.stat().st_size == end - start + 1:
                return
        except Exception as e:  # resume from where the part stopped
            log.warning("range %d-%d attempt %d failed: %r", start, end, attempt, e)
            time.sleep(min(60, 2 ** attempt))
    raise RuntimeError(f"could not download bytes {start}-{end} of {url}")


def md5sum(p, chunk=1 << 24):
    h = hashlib.md5()
    with open(p, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def parallel_download(url, dest, md5=None, connections=32):
    dest = Path(dest)
    if dest.exists() and (md5 is None or md5sum(dest) == md5):
        log.info("%s already present", dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    size = _size(url)
    step = -(-size // connections)
    parts_dir = dest.with_name(dest.name + ".parts")
    parts_dir.mkdir(exist_ok=True)
    ranges = [(i, i * step, min(size, (i + 1) * step) - 1) for i in range(connections) if i * step < size]
    t0 = time.time()
    log.info("downloading %s (%.1f MB) with %d connections", url, size / 1e6, len(ranges))
    with ThreadPoolExecutor(len(ranges)) as ex:
        list(ex.map(lambda r: _fetch_range(url, r[1], r[2], parts_dir / f"{r[0]:04d}"), ranges))
    tmp = dest.with_name(dest.name + ".tmp")
    with open(tmp, "wb") as out:
        for i, _, _ in ranges:
            with open(parts_dir / f"{i:04d}", "rb") as f:
                shutil.copyfileobj(f, out, 1 << 24)
    if md5 is not None and md5sum(tmp) != md5:
        tmp.unlink()
        shutil.rmtree(parts_dir)
        raise RuntimeError(f"md5 mismatch for {url}")
    tmp.rename(dest)
    shutil.rmtree(parts_dir)
    log.info("done %s in %.0fs", dest, time.time() - t0)
    return dest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--connections", type=int, default=32)
    args = ap.parse_args()
    cfg = load_config()
    set_env(cfg)
    import ir_datasets
    info = json.load(open(Path(ir_datasets.__file__).parent / "etc" / "downloads.json"))["msmarco-passage"]
    home = Path(os.environ["IR_DATASETS_HOME"]) / "msmarco-passage"
    for k in IRDS_KEYS:
        parallel_download(info[k]["url"], home / info[k]["cache_path"], info[k]["expected_md5"], args.connections)
    from pyserini.search.lucene import LuceneSearcher   # downloads + unpacks the index (served from HF)
    s = LuceneSearcher.from_prebuilt_index(cfg["data"]["bm25"]["pyserini_index"])
    log.info("pyserini index ready: %d docs", s.num_docs)
    log.info("data under %s", ROOT / "data")


if __name__ == "__main__":
    main()
