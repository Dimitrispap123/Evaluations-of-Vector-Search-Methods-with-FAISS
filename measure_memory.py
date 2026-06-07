import os, io, gc, argparse
import numpy as np
import pandas as pd
import faiss

parser = argparse.ArgumentParser()
parser.add_argument("--data_root",   default="./data")
parser.add_argument("--results_dir", default="./results")
args = parser.parse_args()
os.makedirs(args.results_dir, exist_ok=True)


n_threads = min(os.cpu_count() or 1, 64)
faiss.omp_set_num_threads(n_threads)

K     = 10
NLIST = 4096
EF_CONSTRUCTION = 200
SQ_BITS    = [4, 6, 8]
LSH_NBITS  = [128, 256, 512]


def read_fvecs(path):
    a = np.fromfile(path, dtype="int32")
    d = a[0]
    a = a.reshape(-1, d + 1)
    return np.ascontiguousarray(a[:, 1:]).view("float32")

def l2_normalize(x):
    x = np.ascontiguousarray(x.astype("float32", copy=True))
    faiss.normalize_L2(x)
    return x

def valid_pq_m(d, targets=(8, 16, 32)):
    divisors = [m for m in range(2, d + 1) if d % m == 0]
    chosen = []
    for t in targets:
        m = min(divisors, key=lambda x: (abs(x - t), x))
        if m not in chosen:
            chosen.append(m)
    return sorted(chosen)

def index_bytes(index):
    """Serialized size on disk == memory footprint of the index.

    Why this metric: it is the canonical "what does it cost to store/serve
    this index" number. It includes everything FAISS needs to reconstruct
    the index (codes, codebooks, graph links, inverted lists, metadata)
    and excludes Python/runtime overhead, which is what we want.
    """
    buf = faiss.serialize_index(index)
    return int(buf.nbytes)


def load(name):
    d = args.data_root
    if name == "SIFT":
        xb = read_fvecs(f"{d}/sift/sift_base.fvecs")
        return xb, faiss.METRIC_L2
    if name == "GIST":
        xb = read_fvecs(f"{d}/gist/gist_base.fvecs")
        return xb, faiss.METRIC_L2
    if name == "GloVe":
        xb = l2_normalize(read_fvecs(f"{d}/glove/glove_base.fvecs"))
        return xb, faiss.METRIC_INNER_PRODUCT
    raise ValueError(name)

rows = []

for name in ("SIFT", "GIST", "GloVe"):
    try:
        xb, metric = load(name)
    except Exception as e:
        print(f"{name}: SKIPPED ({e})")
        continue
    d, n = xb.shape[1], xb.shape[0]
    print(f"\n=== {name} d={d} n={n} ===")

    for m in valid_pq_m(d):
        idx = faiss.index_factory(d, f"IVF{NLIST},PQ{m}x8", metric)
        idx.train(xb); idx.add(xb)
        nb = index_bytes(idx)
        rows.append(dict(method="IVFPQ", dataset=name, d=d, n_base=n,
                         m=m, sq_bits=None, nbits=None,
                         index_size_bytes=nb,
                         bytes_per_vector=nb / n))
        print(f"  IVFPQ m={m:2d}  {nb/1e6:7.1f} MB  ({nb/n:7.1f} B/vec)")
        del idx; gc.collect()

    for b in SQ_BITS:
        idx = faiss.index_factory(d, f"HNSW32_SQ{b}", metric)
        try:
            idx.hnsw.efConstruction = EF_CONSTRUCTION
        except AttributeError:
            pass
        idx.train(xb); idx.add(xb)
        nb = index_bytes(idx)
        rows.append(dict(method="HNSWSQ", dataset=name, d=d, n_base=n,
                         m=None, sq_bits=b, nbits=None,
                         index_size_bytes=nb,
                         bytes_per_vector=nb / n))
        print(f"  HNSW SQ{b}    {nb/1e6:7.1f} MB  ({nb/n:7.1f} B/vec)")
        del idx; gc.collect()

    for nbits in LSH_NBITS:
        idx = faiss.IndexLSH(d, nbits)
        idx.train(xb); idx.add(xb)
        nb = index_bytes(idx)
        rows.append(dict(method="LSH", dataset=name, d=d, n_base=n,
                         m=None, sq_bits=None, nbits=nbits,
                         index_size_bytes=nb,
                         bytes_per_vector=nb / n))
        print(f"  LSH nbits={nbits:4d} {nb/1e6:7.1f} MB  ({nb/n:7.1f} B/vec)")
        del idx; gc.collect()

    del xb; gc.collect()

out = os.path.join(args.results_dir, "memory_results.csv")
pd.DataFrame(rows).to_csv(out, index=False)
print(f"\nSaved {len(rows)} rows -> {out}")