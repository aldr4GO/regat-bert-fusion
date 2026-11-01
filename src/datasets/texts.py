"""Query / passage texts for the encoders of a multi-hop dataset (title + text, plan Sec. 3.4 use_title)."""
from tqdm import tqdm

from src.common import data_cache_path, dataset_cfg, load_json


def fmt(title, text, use_title):
    return f"{title}. {text}" if use_title and title else text


def raw_passages(cfg, key, pids):
    """{pid: {"title", "text"}}: closed pools from passages.json, BEIR from the ir_datasets corpus (one pass)."""
    dc = dataset_cfg(cfg, key)
    if dc["kind"] == "closed":
        allp = load_json(data_cache_path(cfg, "passages.json", key))
        return {p: allp[p] for p in pids}
    # open retrieval: one pass over the corpus collects the texts of every candidate of every split, cached as a
    # pickle so the later stages (CE, bi-encoder, title links) do not rescan the 5.2M documents
    import pickle
    cache = data_cache_path(cfg, "candidate_passages.pkl", key)
    if cache.exists():
        with open(cache, "rb") as f:
            allp = pickle.load(f)
    else:
        import ir_datasets
        need = {p for s in ("gat_train", "fusion_train", "val", "test") for pl in cands(cfg, key, s).values() for p in pl}
        allp = {}
        for d in tqdm(ir_datasets.load(dc["ird"]["corpus"]).docs_iter(), desc="beir corpus", mininterval=60):
            if d.doc_id in need:
                allp[d.doc_id] = {"title": d.title, "text": d.text}
        with open(cache, "wb") as f:
            pickle.dump(allp, f, protocol=pickle.HIGHEST_PROTOCOL)
    missing = set(pids) - allp.keys()
    assert not missing, f"{len(missing)} pids missing from the corpus"
    return {p: allp[p] for p in pids}


def passage_texts(cfg, key, pids):
    use_title = cfg["multihop"]["use_title"]
    return {p: fmt(v["title"], v["text"], use_title) for p, v in raw_passages(cfg, key, pids).items()}


def queries(cfg, key):
    return load_json(data_cache_path(cfg, "queries.json", key))


def cands(cfg, key, split):
    return load_json(data_cache_path(cfg, f"cands_{split}.json", key))
