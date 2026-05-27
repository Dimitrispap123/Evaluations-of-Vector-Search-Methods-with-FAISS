# Evaluations of Vector Search Methods with FAISS

M.Sc. in Data and Web Science — Advanced Topics in Databases, Project 2

This project benchmarks three approximate nearest neighbour (ANN) methods — IVFPQ, HNSWSQ, and LSH — across three standard datasets (SIFT, GIST, GloVe) using the FAISS library. We measure Recall@10 and QPS under different parameter settings, analyse query hardness with Relative Contrast, and study robustness under query perturbation.

## Requirements

Python 3.9 or higher. On a CPU-only machine install faiss-cpu. On a GPU node install faiss-gpu instead.

    pip install faiss-cpu numpy pandas matplotlib jupyter

## Datasets

The datasets are not included in the repository due to their size. Download them before running any experiments.

SIFT (128-d, Euclidean): ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz — extract into data/sift/

GIST (960-d, Euclidean): ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz — extract into data/gist/

GloVe (100-d, angular): http://nlp.stanford.edu/data/glove.twitter.27B.zip — extract glove.twitter.27B.100d.txt, convert to fvecs format, and place glove_base.fvecs and glove_query.fvecs into data/glove/

Expected layout after download:

    data/
      sift/sift_base.fvecs
      sift/sift_query.fvecs
      gist/gist_base.fvecs
      gist/gist_query.fvecs
      glove/glove_base.fvecs
      glove/glove_query.fvecs

## Running the experiments

There is one script per method. Each script runs all three datasets and saves its results to results/sweep_{method}.csv.

Quick sanity check (smoke profile, ~50k vectors, finishes in minutes):

    python run_experiment.py --method ivfpq --profile smoke
    python run_experiment.py --method hnsw  --profile smoke
    python run_experiment.py --method lsh   --profile smoke

Full benchmark run:

    python run_experiment.py --method ivfpq --profile full
    python run_experiment.py --method hnsw  --profile full
    python run_experiment.py --method lsh   --profile full

After all three finish, merge the CSVs:

    python merge_results.py

This produces results/sweep_results.csv which the notebook reads for all plots and analysis.

## Running on a cluster (SLURM)

The three job scripts are in the slurm/ folder. Before submitting, open each file and uncomment the partition block that matches your allocation (GPU queue for ivfpq and lsh, high-core CPU queue for hnsw). Then submit all three at once:

    sbatch slurm/slurm_ivfpq.sh
    sbatch slurm/slurm_hnsw.sh
    sbatch slurm/slurm_hnsw.sh

Monitor with:

    squeue -u $USER

When all jobs are done, run merge_results.py as above.

## Reproducing the notebook

Open faiss_vector_search.ipynb. Set DATA_ROOT to ./data and PROFILE to full. The notebook expects results/sweep_results.csv to exist before running the analysis and plotting cells. Run the download and experiment steps first, then open the notebook.

## Project structure

    run_experiment.py       main benchmark script (accepts --method and --profile)
    merge_results.py        merges per-method CSVs into one file
    faiss_vector_search.ipynb  analysis, plots, and discussion
    slurm/                  SLURM job scripts for cluster execution
    data/                   datasets (not tracked by git)
    results/                CSV outputs (not tracked by git)
    plots/                  saved figures (not tracked by git)
    modified_queries/       perturbed query sets (not tracked by git)
    logs/                   SLURM job logs (not tracked by git)
