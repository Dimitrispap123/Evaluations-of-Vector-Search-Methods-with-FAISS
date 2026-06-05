# Evaluation of Vector Search Methods with FAISS

**M.Sc. in Data and Web Science — Advanced Topics in Databases, Project #2**

An experimental study comparing three approximate nearest-neighbour (ANN)
search methods from three different families, using [FAISS](https://github.com/facebookresearch/faiss):

| Method     | Family                                  | FAISS factory          |
|------------|-----------------------------------------|------------------------|
| **IVFPQ**  | inverted file + product quantization    | `IVF{nlist},PQ{m}x8`   |
| **HNSWSQ** | graph-based + scalar quantization       | `HNSW32_SQ{b}`         |
| **LSH**    | hashing                                 | `IndexLSH(d, nbits)`   |

The study covers three datasets — **SIFT** (128-d, Euclidean), **GIST**
(960-d, Euclidean) and **GloVe** (100-d, angular) — and measures **Recall@10**,
**QPS**, **memory footprint**, the **recall–QPS trade-off**, query hardness via
**Relative Contrast (RC)**, and **robustness** under two query perturbation
strategies (Gaussian noise and interpolation toward a non-neighbour).

The benchmark sweeps were run on AUTh's **Aristotelis** HPC cluster
(`rome` and `batch` partitions); the analysis notebook then plots and
discusses the results locally.

---

## 1. Repository layout

```
.
├── faiss_vector_search.ipynb       # the analysis notebook (HTML/PDF exports beside it)
├── README.md                       # this file
│
├── run_experiment.py               # cluster: parameter sweeps (one method at a time)
├── merge_results.py                # cluster/local: merges per-method CSVs
├── run_analysis.py                 # cluster: RC, easy/hard, perturbation, adaptation
├── measure_memory.py               # cluster: serialised-size per (method, dataset, structural)
│
├── slurm/
│   ├── slurm_ivfpq.sh              # IVFPQ sweep   → batch  (16 cores, 8h)
│   ├── slurm_hnsw.sh               # HNSWSQ sweep  → rome   (32 cores, 2 days)
│   ├── slurm_lsh.sh                # LSH sweep     → batch  (8 cores, 3h)
│   ├── slurm_analysis.sh           # post-sweep analysis → rome (32 cores, 6h)
│   └── slurm_memory.sh             # memory measurement  → batch (8 cores, 2h)
│
├── data/                           # raw datasets (not in repo — see §3)
│   ├── sift/   sift_base.fvecs   sift_query.fvecs
│   ├── gist/   gist_base.fvecs   gist_query.fvecs
│   └── glove/  glove_base.fvecs  glove_query.fvecs
│
├── results/                        # CSVs written by the cluster scripts
│   ├── sweep_ivfpq.csv             # IVFPQ recall/QPS sweep         (54 rows)
│   ├── sweep_hnsw.csv              # HNSWSQ recall/QPS sweep        (45 rows)
│   ├── sweep_lsh.csv               # LSH recall/QPS sweep           ( 9 rows)
│   ├── sweep_results.csv           # the three above concatenated   (108 rows)
│   ├── best_configs.csv            # best operating point per (method, dataset)
│   ├── hardness_results.csv        # easy vs hard recall            (18 rows)
│   ├── perturbation_results.csv    # Gaussian + interpolation       (72 rows)
│   ├── adaptation_results.csv      # search-effort sweeps on hard   (33 rows)
│   └── memory_results.csv          # serialised index sizes         (27 rows)
│
├── modified_queries/               # required deliverable — perturbed queries + RC splits
│   ├── sift/    query_sigma{0.00,0.01,0.05,0.10}.fvecs
│   │            query_alpha{0.05,0.10,0.20}.fvecs
│   │            hard_idx.npy   easy_idx.npy   rc_values.npy
│   │            per_query_recall_{IVFPQ,HNSWSQ,LSH}.npy
│   ├── gist/    (same layout)
│   └── glove/   (same layout)
│
├── plots/                          # PNGs the notebook writes
│   ├── tradeoff_{SIFT,GIST,GloVe}.png
│   ├── winner_heatmap.png
│   ├── pareto_frontier.png
│   ├── memory_per_vector.png       (§5.4)
│   ├── rc_distributions.png
│   ├── easy_vs_hard_recall.png
│   ├── rc_recall_scatter.png       (§6.2 — new)
│   ├── perturbation_recall.png     (now has 2 strategies)
│   └── adaptation_hard_queries.png
│
└── logs/                           # SLURM .out/.err — kept as evidence of runs
```

---

## 2. Environment

* Python 3.9 or higher (tested on 3.13).
* CPU-only — no GPU is required (and `IndexLSH` / `HNSW` have no FAISS GPU
  implementation anyway; only IVFPQ has one, but using it complicates the
  install with no meaningful benefit at these dataset sizes).

```bash
pip install --upgrade pip
pip install faiss-cpu numpy pandas matplotlib scipy h5py jupyter nbconvert
```

On Aristotelis, before creating the venv:

```bash
module load gcc/14.2.0 python/3.13.0
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install faiss-cpu numpy pandas matplotlib scipy h5py jupyter nbconvert
```

---

## 3. Downloading the datasets

The datasets are **not in the repository** (~5 GB). All three readers expect
the `.fvecs` format (little-endian `int32` dimension, then `d` `float32`
values per record).

### SIFT (128-d, Euclidean)

```bash
mkdir -p data/sift && cd data/sift
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xzf sift.tar.gz --strip-components=1
rm sift.tar.gz
cd ../..
```

### GIST (960-d, Euclidean)

```bash
mkdir -p data/gist && cd data/gist
wget ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz
tar -xzf gist.tar.gz --strip-components=1
rm gist.tar.gz
cd ../..
```

### GloVe (100-d, angular)

We download the ANN-Benchmarks pre-built HDF5 (instead of parsing the raw
Stanford text), then convert to `.fvecs`:

```bash
mkdir -p data/glove && cd data/glove
wget http://ann-benchmarks.com/glove-100-angular.hdf5
python - <<'PY'
import h5py, numpy as np
def write_fvecs(path, arr):
    arr = np.ascontiguousarray(arr.astype('float32')); n, d = arr.shape
    out = np.empty((n, d+1), dtype='float32')
    out[:, 0] = np.frombuffer(np.int32(d).tobytes(), dtype='float32')[0]
    out[:, 1:] = arr; out.tofile(path)
with h5py.File('glove-100-angular.hdf5', 'r') as f:
    write_fvecs('glove_base.fvecs',  np.array(f['train']))
    write_fvecs('glove_query.fvecs', np.array(f['test']))
PY
rm glove-100-angular.hdf5
cd ../..
```

GloVe's vectors are **L2-normalised internally** by the readers so that
inner-product search equals cosine similarity; you do not need to normalise
them when converting.

---

## 4. Running on the Aristotelis HPC cluster

All heavy computation runs through SLURM. There are five independent jobs;
they write CSVs to `results/` and the analysis script also writes
per-dataset arrays to `modified_queries/`.

### 4.1 Partition choice

| Job             | Partition | CPUs | Memory | Walltime | Rationale                                                |
|-----------------|-----------|------|--------|----------|----------------------------------------------------------|
| IVFPQ sweep     | `batch`   | 16   | 64 GB  | 8 h      | k-means parallelises moderately, modest cores enough     |
| **HNSW sweep**  | `rome`    | 32   | 128 GB | 2 days   | Graph build is OpenMP-heavy; rome's AMD EPYC scales well |
| LSH sweep       | `batch`   | 8    | 32 GB  | 3 h      | LSH is the cheapest of the three                         |
| Analysis        | `rome`    | 32   | 128 GB | 6 h      | RC + perturbation + adaptation, similar cost to HNSW     |
| Memory measure  | `batch`   | 8    | 64 GB  | 2 h      | Pure builds, no search                                   |

`rome` enforces a 60 % CPU-utilisation policy; both jobs above stay above
that threshold (HNSW measured 84 %, analysis 64 %).

### 4.2 First-time setup on the cluster

```bash
ssh <user>@aristotle.it.auth.gr
# upload the project files (or scp from your machine)
cd ~/project

# Fix any Windows line endings on shell scripts and Python files
sed -i 's/\r$//' slurm/*.sh *.py

# Create the venv (one time)
module load gcc/14.2.0 python/3.13.0
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install faiss-cpu numpy pandas matplotlib scipy h5py jupyter nbconvert
deactivate
```

### 4.3 Submitting the jobs

Order does not matter — they are independent.

```bash
sbatch slurm/slurm_ivfpq.sh
sbatch slurm/slurm_hnsw.sh
sbatch slurm/slurm_lsh.sh
sbatch slurm/slurm_analysis.sh
sbatch slurm/slurm_memory.sh

squeue -u $USER                       # monitor
squeue -u $USER --start               # estimated start time for pending jobs
sacct -u $USER -X --starttime=$(date -d '24 hours ago' +%Y-%m-%dT%H:%M)
```

Typical wall-clock times on the recommended partitions:

* LSH sweep — 5 minutes
* HNSW sweep — 22 minutes (on rome)
* IVFPQ sweep — 25 minutes
* Analysis — 25–45 minutes (45 with the interpolation sweep included)
* Memory — 15–25 minutes

Queue wait times depend on cluster load; expect anywhere from instant to
several hours at end of semester.

### 4.4 Merging and verification

After the three sweep jobs finish:

```bash
source venv/bin/activate
python merge_results.py
deactivate
```

This produces `results/sweep_results.csv`. Check resource use of each job
to confirm the rome jobs stayed above 60 % efficiency:

```bash
seff <jobid>
```

---

## 5. Reproducing the notebook

The notebook reads CSVs and `.npy` files from disk; it does **not** re-run
the expensive sweeps. The single "local sweep" cell (§4 of the notebook)
is intentionally commented out — the cluster does that work.

```bash
jupyter notebook faiss_vector_search.ipynb
```

In the configuration cell (cell 2 / §0):

```python
DATA_ROOT = "./data"
PROFILE   = "full"
```

`PROFILE` controls only how the data **loaders** sub-sample — the analysis
cells read the cluster-produced CSVs regardless of this setting. Leave it
on `"full"` to keep the RC and perturbation analyses consistent with the
cluster output.

Then **Kernel → Restart & Run All**. With the cluster CSVs present, the
whole notebook runs in well under a minute.

### Exporting the notebook

```bash
jupyter nbconvert --to html faiss_vector_search.ipynb
jupyter nbconvert --to pdf  faiss_vector_search.ipynb       # needs LaTeX
```

If PDF export fails (no LaTeX), open the HTML in a browser and print to PDF.

---

## 6. Methodology notes

### 6.1 PQ sub-space divisibility

Product Quantization requires the dimension `d` to be divisible by `m`.
The brief suggests `m ∈ {8, 16, 32}`; these all divide SIFT (128) and
GIST (960), but **not GloVe (100)**. `valid_pq_m()` in both
`run_experiment.py` and `run_analysis.py` therefore snaps each target to
the nearest divisor of `d`:

```
d = 128  →  m ∈ {8, 16, 32}
d = 960  →  m ∈ {8, 16, 32}
d = 100  →  m ∈ {10, 20, 25}        # nearest divisors of 100
```

Every dataset gets a comparable 3-point PQ sweep.

### 6.2 Angular GloVe

GloVe is treated as angular: vectors are L2-normalised in the loader, after
which inner-product search equals cosine similarity. RC for GloVe uses the
cosine-distance proxy `d(q, x) = 1 − qᵀx`, a proper distance on the unit
sphere.

### 6.3 Ground truth

Ground truth is **always exact**, produced by a brute-force `IndexFlat`
search (`IndexFlatL2` for SIFT/GIST, `IndexFlatIP` for normalised GloVe).
For the perturbation experiments, ground truth is **recomputed on every
perturbed query set** — we ask "can the index still find the *true*
neighbours of the *noisy* query", which is the meaningful robustness
question.

### 6.4 Two perturbation strategies

The brief suggests two ways to make queries harder; both are implemented:

* **Gaussian noise** (`add_noise`): `q̃ = q + σ·ε`, `ε ~ N(0, I)`. Adds
  isotropic random jitter; recall drops because distance ranking becomes
  noisier. σ is scaled by mean query norm so it is comparable across
  datasets.
* **Interpolation toward a non-neighbour**
  (`interpolate_toward_nonneighbor`): `q' = (1 − α)·q + α·x_far`, where
  `x_far` is a uniformly random base point (not a true 10-NN with
  probability > 0.99999). Systematically lowers RC by construction — this
  is the "harder query" definition the brief gives.

### 6.5 LSH on GPU

`IndexLSH` has no FAISS GPU implementation; passing it through
`index_cpu_to_gpu()` raises. The `build_lsh()` helper deliberately does not
call `maybe_to_gpu()`, and the LSH SLURM script targets a CPU partition.
This was a latent bug fixed before final submission.

### 6.6 Timing

Each search is run three times and the **best** wall-clock is kept (the
ANN-Benchmarks convention) — this reduces noise from background load on
shared nodes. Thread count is fixed per job (`OMP_NUM_THREADS` = SLURM CPU
count) so QPS values are comparable within a job, though not necessarily
between jobs on different partitions.

### 6.7 Memory metric

`measure_memory.py` reports the **serialised index size**
(`faiss.serialize_index(idx).nbytes`). This is the canonical "what does it
cost to store/serve this index" number: it includes everything FAISS needs
to reconstruct the index (codes, codebooks, graph links, inverted lists,
metadata) and excludes Python runtime overhead.

### 6.8 Reproducibility

* `RNG_SEED = 1234` in every script.
* `OMP_NUM_THREADS` is pinned to the SLURM CPU allocation.
* Recall values are bitwise reproducible across runs on identical hardware;
  QPS values reproduce within a few percent due to system noise.

### 6.9 Known minor caveat

In `run_analysis.py`, the saved perturbed-query files in `modified_queries/`
and the perturbation evaluation in `perturbation_results.csv` draw noise
**independently** from the same distribution at each (σ or α). Statistical
properties (mean recall, mean RC) are equivalent, but the saved query
arrays are not bit-identical to the queries actually measured. Mentioned
for transparency; conclusions are not affected.

---

## 7. Deliverables checklist

* `faiss_vector_search.ipynb` — the analysis notebook (executed)
* `faiss_vector_search.html` / `.pdf` — exported versions
* `results/sweep_results.csv` and per-method CSVs
* `results/{hardness,perturbation,adaptation,memory}_results.csv`
* `results/best_configs.csv`
* `modified_queries/{sift,gist,glove}/...` — perturbed queries at multiple
  σ and α values, plus the easy/hard RC splits and per-query recall arrays
* `plots/*.png` — figures used in the poster
* `poster.pdf` — A0 poster
* `README.md` — this file
* `run_experiment.py`, `run_analysis.py`, `measure_memory.py`,
  `merge_results.py` — the four scripts the cluster runs
* `slurm/*.sh` — the five SLURM job scripts

`data/` and `venv/` are not part of the submission — they are recreated
from this README's instructions.