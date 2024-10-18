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
import diskannpy

from heapq import heapify, heappush, heappop 

from collections import defaultdict
from datasketch import MinHash, MinHashLSH, LeanMinHash
from sklearn.cluster import DBSCAN, KMeans
from unionfind import UnionFind

from neurips23.filter.base import BaseFilterANN
from benchmark.datasets import DATASETS
from benchmark.dataset_io import download_accelerated

class Diskann(BaseFilterANN):

    def __init__(self, metric, index_params):
        self.historical_filters_file = index_params['historical_filters_file']
        self.historical_filters_percentage = float(index_params['historical_filters_percentage'])
        self.is_and = ast.literal_eval(index_params['is_and'])
        self.M = int(index_params['M'] )
        self.ef_construction = int(index_params['ef_construction'])
        self.index_budget = int(index_params['index_budget'])
        self.bitvector_cutoff = int(index_params['bitvector_cutoff'])
        self.workload_window_size = int(index_params['workload_window_size'])
        self.heterogeneous_indexing = ast.literal_eval(index_params['heterogeneous_indexing'])
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
        # return os.path.join(index_dir, self.index_name())

    def set_query_arguments(self, query_args):
        self._query_args = query_args
        if 'complexity' in query_args:
            self.complexity = query_args['complexity']
    
    def fit(self, dataset):
        start = time.time()
        ds = DATASETS[dataset]()
        self.dtype = self.translate_dtype(ds.dtype)

        if hasattr(self, 'index'):
            print("Index already exists, skipping fit")
            return

        metadata_arr = ds.get_dataset_metadata()
        print(metadata_arr.shape)
        labels = []
        for i in range(ds.nb):
            labels.append([])
        rows, cols = metadata_arr.nonzero() 
        for row, col in zip(rows, cols):
            labels[row].append(str(col))



        # diskannpy.build_memory_index(
        #     data = ds.get_dataset(),
        #     distance_metric = 'l2',
        #     index_directory = '/home/zl20/bigann/biganntest/diskann_index2',
        #     complexity = 150,
        #     graph_degree = 100,
        #     num_threads = self.num_index_construction_threads,
        #     alpha = 1.2,
        #     filter_labels = labels,
        #     filter_complexity = 90,
        #     index_prefix = 'ann'
        # )

        self.index = diskannpy.StaticMemoryIndex(
            index_directory = '~/bigann/biganntest/diskann_index',
            num_threads = self.num_index_construction_threads,
            initial_search_complexity = 150,
            index_prefix = 'ann',
            distance_metric = 'l2',
            enable_filters = True
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

        metadata_arr = ds.get_dataset_metadata()
        print(metadata_arr.shape)
        labels = []
        for i in range(ds.nb):
            labels.append([])
        rows, cols = metadata_arr.nonzero() 
        for row, col in zip(rows, cols):
            labels[row].append(str(col))

        # diskannpy.build_memory_index(
        #     data = ds.get_dataset(),
        #     distance_metric = 'l2',
        #     index_directory = '/home/zl20/bigann/biganntest/diskann_index2',
        #     complexity = 150,
        #     graph_degree = 100,
        #     num_threads = self.num_index_construction_threads,
        #     alpha = 1.2,
        #     filter_labels = labels,
        #     filter_complexity = 90,
        #     index_prefix = 'ann'
        # )

        self.index = diskannpy.StaticMemoryIndex(
            index_directory = '~/bigann/biganntest/diskann_index',
            num_threads = self.num_index_construction_threads,
            initial_search_complexity = 150,
            index_prefix = 'ann',
            distance_metric = 'l2',
            enable_filters = True
        )

        print("Index initialized")
        print(f"Index fit in {time.time() - start} seconds")
    
    def filtered_query(self, X, filter, k):
        start = time.time()

        rows, cols = filter.nonzero()
        filter_dict = defaultdict(list)

        for row, col in zip(rows, cols):
            filter_dict[row].append(col)

        nq = X.shape[0]
        reses = []
        for i in range(nq):
            reses.append(self.index.search(
                X[i],
                k,
                self.complexity,
                str(filter_dict[i][0])
            ).identifiers)
        self.res = np.array(reses)

        # filters = [None] * len(filter_dict.keys())
        # for i in filter_dict.keys():
        #     filters[i] = hnswlib.QueryFilter(set(filter_dict[i]), self.is_and)

        # print(f"Filter construction took {time.time() - start} seconds")
        # search_start = time.time()
        # nq = X.shape[0]
        # print("ef search:", self.ef_search)
        # self.res = self.index.batch_filter_search(
        #     X,
        #     filters,
        #     nq,
        #     k,
        #     self.ef_search,
        #     1
        # )
        print("result head:")
        print(self.res[:10])
        print(self.res.shape)
        print(f"Search took {time.time() - start} seconds")

    def get_results(self):
        return np.array(self.res)
    
    def __str__(self):
        return f"Diskann"
    
    def index_name(self):
        return f"diskann"
