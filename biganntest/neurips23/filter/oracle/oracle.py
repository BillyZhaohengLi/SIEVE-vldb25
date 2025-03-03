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

from heapq import heapify, heappush, heappop 

from collections import defaultdict
from datasketch import MinHash, MinHashLSH, LeanMinHash
from sklearn.cluster import DBSCAN, KMeans
from unionfind import UnionFind

from neurips23.filter.base import BaseFilterANN
from benchmark.datasets import DATASETS
from benchmark.dataset_io import download_accelerated

class Oracle(BaseFilterANN):

    def __init__(self, metric, index_params):
        self.historical_filters_file = index_params['historical_filters_file']
        self.historical_filters_percentage = float(index_params['historical_filters_percentage'])
        self.is_and = ast.literal_eval(index_params['is_and'])
        self.M = int(index_params['M'] )
        self.ef_construction = int(index_params['ef_construction'])
        self.index_budget = float(index_params['index_budget'])
        self.bitvector_cutoff = int(index_params['bitvector_cutoff'])
        self.workload_window_size = int(index_params['workload_window_size'])
        self.heterogeneous_indexing = ast.literal_eval(index_params['heterogeneous_indexing'])
        self.heterogeneous_search = ast.literal_eval(index_params['heterogeneous_search'])
        self.num_index_construction_threads = int(index_params['num_index_construction_threads'])
        self.ef_search = 10
        self.is_range = False
        if 'is_range' in index_params:
            self.is_range = True
        print("is range:", self.is_range)

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
        if 'ef_search' in query_args:
            self.ef_search = query_args['ef_search']
    
    def fit(self, dataset):
        start = time.time()
        ds = DATASETS[dataset]()
        self.dtype = self.translate_dtype(ds.dtype)
        historical_filters = pickle.load(open(self.historical_filters_file, "rb"))
        
        if not self.is_range:
            rows, cols = historical_filters.nonzero()
            filter_dict = defaultdict(list)
    
            for row, col in zip(rows, cols):
                if row < self.historical_filters_percentage * historical_filters.shape[0]: 
                    filter_dict[row].append(col)
    
            historical_filters_list = []
            for i in filter_dict.keys():
                historical_filters_list.append(hnswlib.QueryFilter(
                    set(filter_dict[i]), self.is_and))

        else:
            historical_filters_list = []
            for i in range(int(historical_filters.shape[0] * self.historical_filters_percentage)):
                historical_filters_list.append(hnswlib.QueryFilter(historical_filters[i], self.is_and))

        if hasattr(self, 'index'):
            print("Index already exists, skipping fit")
            return

        if self.dtype == "uint8":
            self.index = hnswlib.HierarchicalIndexUint8(
                ds.get_dataset_fn(),
                os.path.join(ds.basedir, ds.ds_metadata_fn),
                historical_filters_list,
                ds.nb,
                ds.d,
                self.M,
                self.ef_construction,
                int(self.index_budget * ds.nb),
                self.bitvector_cutoff,
                self.workload_window_size,
                self.heterogeneous_indexing,
                self.heterogeneous_search,
                self.num_index_construction_threads
            )

        if self.dtype == "float":
            self.index = hnswlib.HierarchicalIndexFloat(
                ds.get_dataset_fn(),
                os.path.join(ds.basedir, ds.ds_metadata_fn),
                historical_filters_list,
                ds.nb,
                ds.d,
                self.M,
                self.ef_construction,
                int(self.index_budget * ds.nb),
                self.bitvector_cutoff,
                self.workload_window_size,
                self.heterogeneous_indexing,
                self.heterogeneous_search,
                self.num_index_construction_threads,
                self.is_range
            )

        print("Index initialized")
        print(f"Index fit in {time.time() - start} seconds")

    def load_index(self, dataset):
        start = time.time()
        ds = DATASETS[dataset]()
        self.dtype = self.translate_dtype(ds.dtype)
        historical_filters = pickle.load(open(self.historical_filters_file, "rb"))
        print("dtype:", self.dtype)

        # there's almost certainly a way to do this in less than 0.1s, which costs us ~200 QPS
        if not self.is_range:
            rows, cols = historical_filters.nonzero()
            filter_dict = defaultdict(list)
    
            for row, col in zip(rows, cols):
                if row < self.historical_filters_percentage * historical_filters.shape[0]: 
                    filter_dict[row].append(col)
    
            historical_filters_list = []
            for i in filter_dict.keys():
                historical_filters_list.append(hnswlib.QueryFilter(
                    set(filter_dict[i]), self.is_and))

        else:
            historical_filters_list = []
            for i in range(int(historical_filters.shape[0] * self.historical_filters_percentage)):
                historical_filters_list.append(hnswlib.QueryFilter(historical_filters[i], self.is_and))
            
        if hasattr(self, 'index'):
            print("Index already exists, skipping fit")
            return
        
        if self.dtype == "uint8":
            self.index = hnswlib.HierarchicalIndexUint8(
                ds.get_dataset_fn(),
                os.path.join(ds.basedir, ds.ds_metadata_fn),
                historical_filters_list,
                ds.nb,
                ds.d,
                self.M,
                self.ef_construction,
                int(self.index_budget * ds.nb),
                self.bitvector_cutoff,
                self.workload_window_size,
                self.heterogeneous_indexing,
                self.heterogeneous_search,
                self.num_index_construction_threads
            )
        if self.dtype == "float":
            self.index = hnswlib.HierarchicalIndexFloat(
                ds.get_dataset_fn(),
                os.path.join(ds.basedir, ds.ds_metadata_fn),
                historical_filters_list,
                ds.nb,
                ds.d,
                self.M,
                self.ef_construction,
                int(self.index_budget * ds.nb),
                self.bitvector_cutoff,
                self.workload_window_size,
                self.heterogeneous_indexing,
                self.heterogeneous_search,
                self.num_index_construction_threads,
                self.is_range
            )

        print("Index initialized")
        # self.index.print_stats()
        print(f"Index fit in {time.time() - start} seconds")
    
    def filtered_query(self, X, filter, k):
        start = time.time()
        if not self.is_range:
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
            self.res, times, cardinalities = self.index.batch_filter_search(
                X,
                filters,
                nq,
                k,
                self.ef_search,
                1
            )

            # Compute QPSes
            gt = read_ibin("~/bigann/biganntest/data/paper/paper_gt.ibin")

            # Sort cardinalities into bins
            cardinalities_idx = [(cardinalities[i], i) for i in range(len(cardinalities))]
            cardinalities_idx.sort()

            # Compute QPS/recall of splits
            splits = 5
            points_per_split = int(len(cardinalities) // splits)
            for i in range(splits):
                print("split", i, ":")
                print("split lower cardinality:", cardinalities_idx[int(i * points_per_split)][0])
                print("split upper cardinality:", cardinalities_idx[int((i + 1) * points_per_split - 1)][0])
                total_time = 0
                total_recall = 0
                for j in range(points_per_split):
                    cur_idx = int(i * points_per_split + j)
                    total_time += times[cardinalities_idx[cur_idx][1]]
                    total_recall += len(set(gt[cardinalities_idx[cur_idx][1]]).intersection(self.res[cardinalities_idx[cur_idx][1]]))
                print("QPS:", points_per_split / total_time)
                print("recall:", total_recall / k / points_per_split)
        else:
            filters = [None] * X.shape[0]
            for i in range(X.shape[0]):
                filters[i] = hnswlib.QueryFilter(filter[i], self.is_and)
            print(f"Filter construction took {time.time() - start} seconds")
            search_start = time.time()
            nq = X.shape[0]
            self.res = self.index.batch_filter_search(
                X,
                filters,
                nq,
                k,
                self.ef_search,
                1
            )
        print("result head:")
        print(self.res[:10])
        print(self.res.shape)
        print(f"Search took {time.time() - search_start} seconds")

    def get_results(self):
        return np.array(self.res)
    
    def __str__(self):
        return f"Oracle"
    
    def index_name(self):
        return f"oracle"
