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
```
### 4.4 Merging and verification

After the three sweep jobs finish:

```bash
source venv/bin/activate
python merge_results.py
deactivate
```
This produces `results/sweep_results.csv`.

## 5. Reproducing the notebook

The notebook reads CSVs and `.npy` files from disk; it does **not** re-run
the expensive sweeps.

```bash
jupyter notebook faiss_vector_search.ipynb
```

Then **Kernel → Restart & Run All**. With the cluster CSVs present, the
whole notebook runs in well under a minute.


