import ast
import numpy as np
import os
import pickle
import random
import scipy
import shutil
import time
import xxhash
import hnswlib
from pympler import asizeof

from collections import defaultdict
from datasketch import MinHash, MinHashLSH, LeanMinHash
from sklearn.cluster import DBSCAN, KMeans
from unionfind import UnionFind

from neurips23.filter.base import BaseFilterANN
from benchmark.datasets import DATASETS
from benchmark.dataset_io import download_accelerated

class AcornGamma(BaseFilterANN):

    def __init__(self, metric, index_params):
        self.is_and = ast.literal_eval(index_params['is_and'])
        self.M = int(index_params['M'] )
        self.gamma = int(index_params['gamma'])
        self.m_beta = int(index_params['m_beta'])
        self.bruteforce_selectivity_threshold = float(index_params['bruteforce_selectivity_threshold'])
        self.num_index_construction_threads = int(index_params['num_index_construction_threads'])
        self.ef_search = 10

    def translate_dist_fn(self, metric):
        if metric == 'euclidean':
            return 'Euclidian'
        elif metric == 'ip':
            return 'mips'
        else:
            raise Exception('Invalid metric')
        
    def translate_dtype(self, dtype):
        if 'float32' in str(dtype):
            return 'float'
        return dtype

    def create_index_dir(self, dataset):
        index_dir = os.path.join(os.getcwd(), "data", "indices", "filter")
        os.makedirs(index_dir, mode=0o777, exist_ok=True)
        index_dir = os.path.join(index_dir, 'parlayivf')
        os.makedirs(index_dir, mode=0o777, exist_ok=True)
        index_dir = os.path.join(index_dir, dataset.short_name())
        os.makedirs(index_dir, mode=0o777, exist_ok=True)
        return os.path.join(index_dir, self.index_name())

    def set_query_arguments(self, query_args):
        self._query_args = query_args
        if 'ef_search' in query_args:
            self.ef_search = query_args['ef_search']

    def fit(self, dataset):
        start = time.time()
        ds = DATASETS[dataset]()
        self.dtype = self.translate_dtype(ds.dtype)

        if hasattr(self, 'index'):
            print("Index already exists, skipping fit")
            return

        if self.dtype == "uint8":
            self.index = hnswlib.AcornIndexUint8(
                ds.get_dataset_fn(),
                os.path.join(ds.basedir, ds.ds_metadata_fn),
                ds.nb,
                ds.d,
                self.M,
                self.gamma,
                self.m_beta,
                self.bruteforce_selectivity_threshold,
                self.num_index_construction_threads
            )

        if self.dtype == "float":
            self.index = hnswlib.AcornIndexFloat(
                ds.get_dataset_fn(),
                os.path.join(ds.basedir, ds.ds_metadata_fn),
                ds.nb,
                ds.d,
                self.M,
                self.gamma,
                self.m_beta,
                self.bruteforce_selectivity_threshold,
                self.num_index_construction_threads
            )
            
        print("Index initialized")
        print(f"Index fit in {time.time() - start} seconds")

    def load_index(self, dataset):
        start = time.time()
        ds = DATASETS[dataset]()
        self.dtype = self.translate_dtype(ds.dtype)

        if hasattr(self, 'index'):
            print("Index already exists, skipping fit")
            return

        if self.dtype == "uint8":
            self.index = hnswlib.AcornIndexUint8(
                ds.get_dataset_fn(),
                os.path.join(ds.basedir, ds.ds_metadata_fn),
                ds.nb,
                ds.d,
                self.M,
                self.gamma,
                self.m_beta,
                self.bruteforce_selectivity_threshold,
                self.num_index_construction_threads
            )

        if self.dtype == "float":
            self.index = hnswlib.AcornIndexFloat(
                ds.get_dataset_fn(),
                os.path.join(ds.basedir, ds.ds_metadata_fn),
                ds.nb,
                ds.d,
                self.M,
                self.gamma,
                self.m_beta,
                self.bruteforce_selectivity_threshold,
                self.num_index_construction_threads
            )
        print("Index initialized")
        print(f"Index fit in {time.time() - start} seconds")


    def set_query_arguments(self, query_args):
        self._query_args = query_args
        if 'ef_search' in query_args:
            self.ef_search = query_args['ef_search']

    def filtered_query(self, X, filter, k):
        start = time.time()

        rows, cols = filter.nonzero()
        filter_dict = defaultdict(list)

        for row, col in zip(rows, cols):
            filter_dict[row].append(col)

        filters = [None] * X.shape[0]
        for i in range(X.shape[0]):
            if i in filter_dict.keys():
                filters[i] = hnswlib.QueryFilter(set(filter_dict[i]), self.is_and)
            else:
                filters[i] = hnswlib.QueryFilter(set(), self.is_and)

        print(f"Filter construction took {time.time() - start} seconds")
        search_start = time.time()
        nq = X.shape[0]
        print("ef search:", self.ef_search)
        self.res = self.index.batch_filter_search(
            X,
            filters,
            nq,
            k,
            self.ef_search,
            1
        )
        # print("result head:")
        # print(self.res[:10])
        # print(self.res.shape)
        print(f"Search took {time.time() - search_start} seconds")

    def get_results(self):
        # print(self.res.shape)
        # print(self.query_dists.shape)
        # print(self.res[:10, :10])
        # print(self.query_dists[:10, :10])
        return np.array(self.res)
    
    def __str__(self):
        return f"AcornGamma"
    
    def index_name(self):
        return f"acorngamma"
