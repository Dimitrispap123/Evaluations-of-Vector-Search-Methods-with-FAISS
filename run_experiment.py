import os, time, gc, argparse, json
import numpy as np
import pandas as pd
import faiss

parser = argparse.ArgumentParser()
parser.add_argument("--method",  required=True, choices=["ivfpq", "hnsw", "lsh"])
parser.add_argument("--profile", default="full", choices=["smoke", "full"])
parser.add_argument("--data_root", default="./data")
parser.add_argument("--results_dir", default="./results")
args = parser.parse_args()

os.makedirs(args.results_dir, exist_ok=True)

USE_GPU = False
if args.method in ("ivfpq", "lsh"):          
    try:
        ngpu = faiss.get_num_gpus()
        if ngpu > 0:
            USE_GPU = True
            print(f"GPU mode: {ngpu} GPU(s) detected — using GPU 0")
        else:
            print("No GPUs detected — using CPU")
    except AttributeError:
        print("faiss-cpu installed (no GPU support) — using CPU")
else:
    print(f"Method {args.method.upper()} is CPU-only (HNSW has no FAISS GPU support)")

n_threads = min(os.cpu_count() or 1, 64)
faiss.omp_set_num_threads(n_threads)
print(f"OpenMP threads : {faiss.omp_get_max_threads()}")
print(f"Method         : {args.method.upper()}")
print(f"Profile        : {args.profile}")


RNG_SEED = 1234
np.random.seed(RNG_SEED)
K = 10

PROFILES = {
    "smoke": dict(max_base=50_000,  max_query=200,  nlist=1024),
    "full":  dict(max_base=None,    max_query=1000, nlist=4096),
}
CFG = PROFILES[args.profile]


def read_fvecs(path, max_rows=None):
    a = np.fromfile(path, dtype="int32")
    if a.size == 0:
        raise ValueError(f"empty file: {path}")
    d = a[0]
    a = a.reshape(-1, d + 1)
    if max_rows:
        a = a[:max_rows]
    return np.ascontiguousarray(a[:, 1:]).view("float32")

def read_bvecs(path, max_rows=None):
    a = np.fromfile(path, dtype="uint8")
    d = a[:4].view("int32")[0]
    a = a.reshape(-1, d + 4)
    if max_rows:
        a = a[:max_rows]
    return np.ascontiguousarray(a[:, 4:]).astype("float32")

def l2_normalize(x):
    x = np.ascontiguousarray(x.astype("float32", copy=True))
    faiss.normalize_L2(x)
    return x

def load_sift():
    d = args.data_root
    return (read_fvecs(f"{d}/sift/sift_base.fvecs",  CFG["max_base"]),
            read_fvecs(f"{d}/sift/sift_query.fvecs", CFG["max_query"]))

def load_gist():
    d = args.data_root
    return (read_fvecs(f"{d}/gist/gist_base.fvecs",  CFG["max_base"]),
            read_fvecs(f"{d}/gist/gist_query.fvecs", CFG["max_query"]))

def load_glove():
    d = args.data_root
    base  = read_fvecs(f"{d}/glove/glove_base.fvecs",  CFG["max_base"])
    query = read_fvecs(f"{d}/glove/glove_query.fvecs", CFG["max_query"])
    return l2_normalize(base), l2_normalize(query)

DATASETS = {
    "SIFT":  dict(loader=load_sift,  metric=faiss.METRIC_L2),
    "GIST":  dict(loader=load_gist,  metric=faiss.METRIC_L2),
    "GloVe": dict(loader=load_glove, metric=faiss.METRIC_INNER_PRODUCT),
}


def build_ground_truth(xb, xq, k, metric):
    idx = (faiss.IndexFlatIP(xb.shape[1]) if metric == faiss.METRIC_INNER_PRODUCT
           else faiss.IndexFlatL2(xb.shape[1]))
    idx.add(xb)
    return idx.search(xq, k)

def recall_at_k(I_approx, I_gt, k):
    hits = sum(
        len(set(I_approx[i, :k]).intersection(set(I_gt[i, :k])))
        for i in range(I_gt.shape[0])
    )
    return hits / (I_gt.shape[0] * k)

def benchmark_index(index, xq, k, repeats=3):
    best_t, D, I = float("inf"), None, None
    for _ in range(repeats):
        t0 = time.perf_counter()
        D_, I_ = index.search(xq, k)
        t1 = time.perf_counter()
        if (t1 - t0) < best_t:
            best_t, D, I = (t1 - t0), D_, I_
    return D, I, best_t, xq.shape[0] / best_t

def set_param(index, name, value):
    faiss.ParameterSpace().set_index_parameter(index, name, value)

def maybe_to_gpu(index):
    """Move index to GPU if available and method supports it."""
    if USE_GPU:
        res = faiss.StandardGpuResources()
        return faiss.index_cpu_to_gpu(res, 0, index)
    return index


def valid_pq_m(d, targets=(8, 16, 32)):
    divisors = [m for m in range(2, d + 1) if d % m == 0]
    chosen = []
    for t in targets:
        m = min(divisors, key=lambda x: (abs(x - t), x))
        if m not in chosen:
            chosen.append(m)
    return sorted(chosen)


NPROBE_VALUES   = [1, 4, 8, 16, 32, 64]
EFSEARCH_VALUES = [16, 32, 64, 128, 256]
SQ_BITS         = [4, 6, 8]
LSH_NBITS       = [128, 256, 512]
EF_CONSTRUCTION = 200


def build_ivfpq(xb, m, metric, nlist):
    d = xb.shape[1]
    index = faiss.index_factory(d, f"IVF{nlist},PQ{m}x8", metric)
    if not index.is_trained:
        index.train(xb)
    index.add(xb)
    return maybe_to_gpu(index)

def build_hnsw(xb, b, metric):
    d = xb.shape[1]
    index = faiss.index_factory(d, f"HNSW32_SQ{b}", metric)
    try:
        index.hnsw.efConstruction = EF_CONSTRUCTION
    except AttributeError:
        pass
    if not index.is_trained:
        index.train(xb)
    index.add(xb)
    return index   

def build_lsh(xb, nbits, metric):
    d = xb.shape[1]
    index = faiss.IndexLSH(d, nbits)
    index.train(xb)
    index.add(xb)
    return index


def sweep_ivfpq(name, xb, xq, metric, I_gt):
    rows = []
    d = xb.shape[1]
    for m in valid_pq_m(d):
        t0 = time.perf_counter()
        idx = build_ivfpq(xb, m, metric, CFG["nlist"])
        build_t = time.perf_counter() - t0
        for nprobe in NPROBE_VALUES:
            set_param(idx, "nprobe", nprobe)
            D, I, st, qps = benchmark_index(idx, xq, K)
            rows.append(dict(
                method="IVFPQ", dataset=name, d=d,
                param1=f"m={m}", param2=f"nprobe={nprobe}",
                m=m, nprobe=nprobe, sq_bits=None, nbits=None,
                recall=recall_at_k(I, I_gt, K), qps=qps, build_time=build_t,
            ))
        print(f"  IVFPQ m={m:2d}  build={build_t:.1f}s  sweep done")
        del idx; gc.collect()
    return rows

def sweep_hnsw(name, xb, xq, metric, I_gt):
    rows = []
    d = xb.shape[1]
    for b in SQ_BITS:
        t0 = time.perf_counter()
        idx = build_hnsw(xb, b, metric)
        build_t = time.perf_counter() - t0
        for ef in EFSEARCH_VALUES:
            set_param(idx, "efSearch", ef)
            D, I, st, qps = benchmark_index(idx, xq, K)
            rows.append(dict(
                method="HNSWSQ", dataset=name, d=d,
                param1=f"SQ{b}", param2=f"efSearch={ef}",
                m=None, nprobe=None, sq_bits=b, nbits=None,
                recall=recall_at_k(I, I_gt, K), qps=qps, build_time=build_t,
            ))
        print(f"  HNSW SQ{b}  build={build_t:.1f}s  sweep done")
        del idx; gc.collect()
    return rows

def sweep_lsh(name, xb, xq, metric, I_gt):
    rows = []
    d = xb.shape[1]
    for nbits in LSH_NBITS:
        t0 = time.perf_counter()
        idx = build_lsh(xb, nbits, metric)
        build_t = time.perf_counter() - t0
        D, I, st, qps = benchmark_index(idx, xq, K)
        rows.append(dict(
            method="LSH", dataset=name, d=d,
            param1=f"nbits={nbits}", param2="-",
            m=None, nprobe=None, sq_bits=None, nbits=nbits,
            recall=recall_at_k(I, I_gt, K), qps=qps, build_time=build_t,
        ))
        print(f"  LSH nbits={nbits}  build={build_t:.1f}s")
        del idx; gc.collect()
    return rows

SWEEPERS = {"ivfpq": sweep_ivfpq, "hnsw": sweep_hnsw, "lsh": sweep_lsh}


all_rows = []

for ds_name, spec in DATASETS.items():
    try:
        xb, xq = spec["loader"]()
        metric  = spec["metric"]
    except Exception as e:
        print(f"{ds_name}: SKIPPED ({e})")
        continue

    print(f"\n=== {ds_name}  base={xb.shape}  query={xq.shape} ===")

    t0 = time.perf_counter()
    _, I_gt = build_ground_truth(xb, xq, K, metric)
    print(f"  Ground truth built in {time.perf_counter()-t0:.1f}s")

    rows = SWEEPERS[args.method](ds_name, xb, xq, metric, I_gt)
    all_rows.extend(rows)

    del xb, xq, I_gt; gc.collect()


out_path = os.path.join(args.results_dir, f"sweep_{args.method}.csv")
df = pd.DataFrame(all_rows)
df.to_csv(out_path, index=False)
print(f"\nSaved {len(df)} rows -> {out_path}")
print(df[["method", "dataset", "param1", "param2", "recall", "qps"]].to_string(index=False))
