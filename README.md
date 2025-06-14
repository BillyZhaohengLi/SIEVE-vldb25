# Repository for SIEVE: Effective Filtered Vector Search with Collection of Indexes

## Project structure:
* `hnswtest`: clone of hnswlib, contains SIEVE's implementation
* `biganntest`: clone of Neurips'23 BigANN benchmark suite, contains experiment setups
* `data_query_attr_generators`: our code for generating data attributes and query filters for the 5 datasets (Paper, UQV, GIST, Sift, MSONG).

## Instructions:
### Setting Up SIEVE

SIEVE contains Python bindings for itself and various baselines.
1. Install ACORN following instructions in their [repository](https://github.com/TAG-Research/ACORN)
2. install SIEVE:
```
cd hnswtest
pip install .
```

### Getting Datasets
1. The yfcc-10M dataset is included with the Neurips'23 BigANN benchmark suite. The other 5 datasets (Paper, UQV, GIST, Sift, MSONG) can be downloaded from the [NHQ repository](https://github.com/YujianFu97/NHQ).
2. Run the Python scripts in `data_query_attr_generators` to generate data attributes and query filters.
3. Place the generated files (and the data) under the `biganntest/data` directory in appropriate subdirectories (e.g., `biganntest/data/sift` for SIFT).

### Running Experiments
Follow the instructions in the Neurips'23 BigANN benchmark suite README.md to run experiments. All methods except for IVF2 do not need to be built into dockerfiles, and should be run with the `--nodocker` flag.
