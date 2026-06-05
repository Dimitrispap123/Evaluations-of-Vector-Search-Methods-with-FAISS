"""
run_analysis.py
Runs the post-sweep analysis (Relative Contrast, easy/hard split, perturbation,
adaptation) for all three datasets and writes the result CSVs + modified queries
that the notebook expects.

This corresponds to §6, §7, §8, §9 of the notebook. It exists as a script so
the heavy work can be run as a SLURM job on the rome partition instead of
inside an interactive Jupyter session.

Usage:
    python run_analysis.py --data_root ./data --results_dir ./results

Outputs:
    results/hardness_results.csv
    results/perturbation_results.csv
    results/adaptation_results.csv
    modified_queries/{sift,gist,glove}/  (perturbed queries + RC splits)
"""

import os, time, gc, argparse
import numpy as np
import pandas as pd
import faiss

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--data_root",   default="./data")
parser.add_argument("--results_dir", default="./results")
parser.add_argument("--query_dir",   default="./modified_queries")
args = parser.parse_args()

os.makedirs(args.results_dir, exist_ok=True)
os.makedirs(args.query_dir,   exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────

RNG_SEED = 1234
np.random.seed(RNG_SEED)

K     = 10
NLIST = 4096       # IVF cells — matches "full" profile in run_experiment.py

n_threads = min(os.cpu_count() or 1, 64)
faiss.omp_set_num_threads(n_threads)
print(f"FAISS threads: {faiss.omp_get_max_threads()}")
print(f"FAISS version: {faiss.__version__}")

# Representative ("middle of the road") config per method — used for both
# easy/hard analysis and perturbation. Adaptation sweeps on top of this.
REP_CONFIG = dict(
    IVFPQ  = dict(m=16, nprobe=16),
    HNSWSQ = dict(sq_bits=8, efSearch=64),
    LSH    = dict(nbits=256),
)

EF_CONSTRUCTION = 200
NPROBE_VALUES   = [1, 4, 8, 16, 32, 64]
EFSEARCH_VALUES = [16, 32, 64, 128, 256]
NOISE_LEVELS    = [0.00, 0.01, 0.05, 0.10]

# ── Dataset loaders ───────────────────────────────────────────────────────────

def read_fvecs(path, max_rows=None):
    a = np.fromfile(path, dtype="int32")
    d = a[0]
    a = a.reshape(-1, d + 1)
    if max_rows:
        a = a[:max_rows]
    return np.ascontiguousarray(a[:, 1:]).view("float32")

def l2_normalize(x):
    x = np.ascontiguousarray(x.astype("float32", copy=True))
    faiss.normalize_L2(x)
    return x

def load_sift():
    return (read_fvecs(f"{args.data_root}/sift/sift_base.fvecs"),
            read_fvecs(f"{args.data_root}/sift/sift_query.fvecs"))

def load_gist():
    return (read_fvecs(f"{args.data_root}/gist/gist_base.fvecs"),
            read_fvecs(f"{args.data_root}/gist/gist_query.fvecs"))

def load_glove():
    base  = read_fvecs(f"{args.data_root}/glove/glove_base.fvecs")
    query = read_fvecs(f"{args.data_root}/glove/glove_query.fvecs")
    return l2_normalize(base), l2_normalize(query)

DATASETS = {
    "SIFT":  dict(loader=load_sift,  metric=faiss.METRIC_L2),
    "GIST":  dict(loader=load_gist,  metric=faiss.METRIC_L2),
    "GloVe": dict(loader=load_glove, metric=faiss.METRIC_INNER_PRODUCT),
}

# ── Eval helpers ──────────────────────────────────────────────────────────────

def build_ground_truth(xb, xq, k, metric):
    idx = (faiss.IndexFlatIP(xb.shape[1]) if metric == faiss.METRIC_INNER_PRODUCT
           else faiss.IndexFlatL2(xb.shape[1]))
    idx.add(xb)
    return idx.search(xq, k)

def recall_at_k(I_approx, I_gt, k):
    hits = sum(len(set(I_approx[i, :k]).intersection(set(I_gt[i, :k])))
               for i in range(I_gt.shape[0]))
    return hits / (I_gt.shape[0] * k)

def benchmark(index, xq, k, repeats=3):
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

def valid_pq_m(d, targets=(8, 16, 32)):
    divisors = [m for m in range(2, d + 1) if d % m == 0]
    chosen = []
    for t in targets:
        m = min(divisors, key=lambda x: (abs(x - t), x))
        if m not in chosen:
            chosen.append(m)
    return sorted(chosen)

# ── Index builders ────────────────────────────────────────────────────────────

def build_ivfpq_rep(xb, metric):
    """Build IVFPQ at the representative config for this dataset."""
    d = xb.shape[1]
    m = REP_CONFIG["IVFPQ"]["m"]
    m = m if d % m == 0 else valid_pq_m(d)[0]
    idx = faiss.index_factory(d, f"IVF{NLIST},PQ{m}x8", metric)
    idx.train(xb)
    idx.add(xb)
    set_param(idx, "nprobe", REP_CONFIG["IVFPQ"]["nprobe"])
    return idx

def build_hnswsq_rep(xb, metric):
    d = xb.shape[1]
    b = REP_CONFIG["HNSWSQ"]["sq_bits"]
    idx = faiss.index_factory(d, f"HNSW32_SQ{b}", metric)
    try:
        idx.hnsw.efConstruction = EF_CONSTRUCTION
    except AttributeError:
        pass
    idx.train(xb)
    idx.add(xb)
    set_param(idx, "efSearch", REP_CONFIG["HNSWSQ"]["efSearch"])
    return idx

def build_lsh_rep(xb, metric):
    d = xb.shape[1]
    idx = faiss.IndexLSH(d, REP_CONFIG["LSH"]["nbits"])
    idx.train(xb)
    idx.add(xb)
    return idx

BUILDERS = {
    "IVFPQ":  build_ivfpq_rep,
    "HNSWSQ": build_hnswsq_rep,
    "LSH":    build_lsh_rep,
}

# ── Relative Contrast ─────────────────────────────────────────────────────────

def compute_rc(xb, xq, k, metric, batch=256):
    d = xb.shape[1]
    gt = (faiss.IndexFlatIP(d) if metric == faiss.METRIC_INNER_PRODUCT
          else faiss.IndexFlatL2(d))
    gt.add(xb)
    Dk, _ = gt.search(xq, k)

    if metric == faiss.METRIC_L2:
        rk = np.sqrt(np.maximum(Dk[:, k - 1], 0.0))
    else:
        rk = 1.0 - Dk[:, k - 1]

    dmean = np.empty(xq.shape[0], dtype="float64")
    for i in range(0, xq.shape[0], batch):
        q = xq[i:i + batch]
        if metric == faiss.METRIC_L2:
            xb2 = (xb ** 2).sum(1)[None, :]
            q2  = (q ** 2).sum(1)[:, None]
            dist = np.sqrt(np.maximum(xb2 + q2 - 2.0 * q @ xb.T, 0.0))
        else:
            dist = 1.0 - q @ xb.T
        dmean[i:i + batch] = dist.mean(axis=1)

    rk = np.where(rk <= 1e-12, 1e-12, rk)
    return dmean / rk

def split_easy_hard(rc, frac=0.25):
    n = len(rc)
    order = np.argsort(rc)
    cut = max(1, int(round(frac * n)))
    return order[:cut], order[-cut:]

# ── Perturbation ──────────────────────────────────────────────────────────────

def add_noise(xq, sigma_rel, metric, rng=None):
    # Same Gaussian perturbation as before, but now accepts an explicit RNG.
    # Why: previously we used the global np.random state, so the noise written
    # to disk in §9 and the noise evaluated in §7 were DIFFERENT draws at the
    # same sigma. Now the caller passes a deterministic Generator seeded by
    # (dataset, sigma), so "saved queries" == "measured queries". Reproducible.
    if rng is None:
        rng = np.random                     # back-compat fallback
    mean_norm = float(np.linalg.norm(xq, axis=1).mean())
    sigma = sigma_rel * mean_norm
    noisy = (xq + sigma * rng.standard_normal(xq.shape).astype("float32")
            ).astype("float32") if hasattr(rng, "standard_normal") else \
            (xq + sigma * rng.randn(*xq.shape).astype("float32")).astype("float32")
    if metric == faiss.METRIC_INNER_PRODUCT:
        noisy = l2_normalize(noisy)
    return np.ascontiguousarray(noisy)


def make_rng(dataset_name, sigma, strategy="gaussian"):
    # Stable seed derived from (dataset, sigma, strategy). Same inputs ->
    # same noise, regardless of execution order or other random draws.
    key = f"{dataset_name}|{strategy}|{sigma:.4f}".encode()
    seed = int(np.frombuffer(
        __import__("hashlib").sha256(key).digest()[:8], dtype=np.uint64)[0]
        & 0x7FFFFFFF)
    return np.random.default_rng(seed)

def interpolate_toward_nonneighbor(xb, xq, alpha, metric, dataset_name):
    rng = make_rng(dataset_name, alpha, strategy="interp")
    n, d = xq.shape
    # Random non-neighbour index for each query. With ~1M base vectors and
    # only 10 neighbours, a uniform sample is a non-neighbour with prob > 0.99999.
    far_idx = rng.integers(0, xb.shape[0], size=n)
    x_far   = xb[far_idx]
    qprime  = ((1.0 - alpha) * xq + alpha * x_far).astype("float32")
    if metric == faiss.METRIC_INNER_PRODUCT:
        qprime = l2_normalize(qprime)
    return np.ascontiguousarray(qprime)

def write_fvecs(path, arr):
    arr = np.ascontiguousarray(arr.astype("float32"))
    n, d = arr.shape
    out = np.empty((n, d + 1), dtype="float32")
    out[:, 0] = np.frombuffer(np.int32(d).tobytes(), dtype="float32")[0]
    out[:, 1:] = arr
    out.tofile(path)

# ── Main loop ─────────────────────────────────────────────────────────────────

hardness_rows   = []
perturb_rows    = []
adaptation_rows = []
RC = {}

for name, spec in DATASETS.items():
    print(f"\n========= {name} =========", flush=True)
    t0_ds = time.perf_counter()

    xb, xq = spec["loader"]()
    metric = spec["metric"]
    d = xb.shape[1]
    print(f"  loaded base={xb.shape}  query={xq.shape}  d={d}", flush=True)

    # ── Compute RC ────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    rc = compute_rc(xb, xq, K, metric)
    RC[name] = rc
    print(f"  RC computed in {time.perf_counter()-t0:.1f}s  "
          f"(min={rc.min():.3f}  median={np.median(rc):.3f}  max={rc.max():.3f})",
          flush=True)

    hard_idx, easy_idx = split_easy_hard(rc)

# ── Save modified queries (§9) — generate ONCE, reuse for evaluation ─────
    # Why this is better than before:
    #   - Old code called add_noise() here, then AGAIN in the perturbation
    #     loop -> different random draws for the SAME sigma. The files on
    #     disk did not correspond to the numbers in perturbation_results.csv.
    #   - Now we materialise every (dataset, sigma) noisy query set ONCE,
    #     using a seed derived from (dataset, sigma), and reuse the array
    #     downstream. Saved queries == measured queries. Fully reproducible.
    folder = os.path.join(args.query_dir, name.lower())
    os.makedirs(folder, exist_ok=True)

    noisy_queries = {}                                    # gaussian variants
    interp_queries = {}                                   # interpolation variants
    for sigma in NOISE_LEVELS:
        if sigma == 0.0:
            noisy_queries[sigma] = xq                     # clean baseline
        else:
            rng = make_rng(name, sigma, strategy="gaussian")
            noisy_queries[sigma] = add_noise(xq, sigma, metric, rng=rng)
        write_fvecs(os.path.join(folder, f"query_sigma{sigma:.2f}.fvecs"),
                    noisy_queries[sigma])

    # Interpolation perturbation: alpha is dimensionless, fraction toward a
    # random non-neighbour. We mirror the Gaussian sigma grid for symmetry
    # in the §7 plot.
    INTERP_ALPHAS = [0.00, 0.05, 0.10, 0.20]              # 0 == clean
    for alpha in INTERP_ALPHAS:
        if alpha == 0.0:
            interp_queries[alpha] = xq
        else:
            interp_queries[alpha] = interpolate_toward_nonneighbor(
                xb, xq, alpha, metric, dataset_name=name)
        write_fvecs(os.path.join(folder, f"query_interp{alpha:.2f}.fvecs"),
                    interp_queries[alpha])

    np.save(os.path.join(folder, "hard_idx.npy"),  hard_idx)
    np.save(os.path.join(folder, "easy_idx.npy"),  easy_idx)
    np.save(os.path.join(folder, "rc_values.npy"), rc)
    print(f"  modified queries written -> {folder}", flush=True)

    # ── Build each index ONCE, reuse for all subsections ──────────────────────
    indexes = {}
    for method in ("IVFPQ", "HNSWSQ", "LSH"):
        t0 = time.perf_counter()
        indexes[method] = BUILDERS[method](xb, metric)
        print(f"  {method} representative index built in "
              f"{time.perf_counter()-t0:.1f}s", flush=True)

    # ── §6.1 Easy vs hard recall + per-query recall for RC-correlation ───────
    # Why per-query recall: the brief asks "is RC a useful predictor of search
    # difficulty?". The cleanest answer is Spearman correlation between
    # per-query RC and per-query recall (one number, plus a scatter). We
    # save the arrays now while indexes are in memory; the notebook just
    # loads them and computes ρ.
    print("  -- easy vs hard + per-query recall --", flush=True)
    _, I_gt_full = build_ground_truth(xb, xq, K, metric)
    for method, idx in indexes.items():
        D, I, _, qps_full = benchmark(idx, xq, K)
        # Per-query recall@K for this (dataset, method) at the rep config.
        per_q = np.array([
            len(set(I[i, :K]).intersection(I_gt_full[i, :K])) / K
            for i in range(I_gt_full.shape[0])
        ], dtype="float64")
        np.save(os.path.join(folder, f"per_query_recall_{method}.npy"), per_q)
    for grp, idx_subset in (("hard", hard_idx), ("easy", easy_idx)):
        xq_sub = np.ascontiguousarray(xq[idx_subset])
        _, I_gt = build_ground_truth(xb, xq_sub, K, metric)
        for method, idx in indexes.items():
            D, I, _, qps = benchmark(idx, xq_sub, K)
            hardness_rows.append(dict(
                dataset=name, method=method, group=grp,
                recall=recall_at_k(I, I_gt, K), qps=qps,
                mean_rc=float(rc[idx_subset].mean()),
            ))
        print(f"    group={grp:4s} done", flush=True)

    # ── §7 Perturbation — Gaussian AND interpolation, in one CSV ─────────────
    # Why the schema change: we now emit a `strategy` column ("gaussian" or
    # "interp") and unify the sigma/alpha axis under a single "level" column.
    # The notebook will facet by strategy.
    print("  -- perturbation (gaussian + interp) --", flush=True)

    def _eval_perturb(xq_var, strategy, level):
        _, I_gt = build_ground_truth(xb, xq_var, K, metric)
        mean_rc_var = float(compute_rc(xb, xq_var, K, metric).mean())
        for method, idx in indexes.items():
            D, I, _, qps = benchmark(idx, xq_var, K)
            perturb_rows.append(dict(
                dataset=name, method=method, strategy=strategy,
                level=level, recall=recall_at_k(I, I_gt, K), qps=qps,
                mean_rc=mean_rc_var,
            ))
        print(f"    {strategy} level={level:.2f}  mean_RC={mean_rc_var:.3f}",
              flush=True)

    for sigma in NOISE_LEVELS:
        _eval_perturb(noisy_queries[sigma], "gaussian", sigma)
    for alpha in INTERP_ALPHAS:
        _eval_perturb(interp_queries[alpha], "interp", alpha)
    # ── §8 Adaptation (sweep on hard subset) ─────────────────────────────────
    print("  -- adaptation --", flush=True)
    xq_hard = np.ascontiguousarray(xq[hard_idx])
    _, I_gt_hard = build_ground_truth(xb, xq_hard, K, metric)

    # IVFPQ — sweep nprobe on the existing index
    for nprobe in NPROBE_VALUES:
        set_param(indexes["IVFPQ"], "nprobe", nprobe)
        D, I, _, qps = benchmark(indexes["IVFPQ"], xq_hard, K)
        adaptation_rows.append(dict(
            dataset=name, method="IVFPQ", knob="nprobe", value=nprobe,
            recall=recall_at_k(I, I_gt_hard, K), qps=qps,
        ))
    set_param(indexes["IVFPQ"], "nprobe", REP_CONFIG["IVFPQ"]["nprobe"])  # restore

    # HNSWSQ — sweep efSearch
    for ef in EFSEARCH_VALUES:
        set_param(indexes["HNSWSQ"], "efSearch", ef)
        D, I, _, qps = benchmark(indexes["HNSWSQ"], xq_hard, K)
        adaptation_rows.append(dict(
            dataset=name, method="HNSWSQ", knob="efSearch", value=ef,
            recall=recall_at_k(I, I_gt_hard, K), qps=qps,
        ))
    print(f"  adaptation done", flush=True)

    # ── Cleanup ──────────────────────────────────────────────────────────────
    del indexes, xb, xq
    gc.collect()
    print(f"  {name} total time: {time.perf_counter()-t0_ds:.1f}s", flush=True)

# ── Save outputs ──────────────────────────────────────────────────────────────

pd.DataFrame(hardness_rows  ).to_csv(f"{args.results_dir}/hardness_results.csv",     index=False)
pd.DataFrame(perturb_rows   ).to_csv(f"{args.results_dir}/perturbation_results.csv", index=False)
pd.DataFrame(adaptation_rows).to_csv(f"{args.results_dir}/adaptation_results.csv",   index=False)

print("\nAll analyses complete.")
print(f"  -> {args.results_dir}/hardness_results.csv     ({len(hardness_rows)} rows)")
print(f"  -> {args.results_dir}/perturbation_results.csv ({len(perturb_rows)} rows)")
print(f"  -> {args.results_dir}/adaptation_results.csv   ({len(adaptation_rows)} rows)")
print(f"  -> {args.query_dir}/{{sift,gist,glove}}/...")
