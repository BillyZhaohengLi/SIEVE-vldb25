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

class MyMethod(BaseFilterANN):

    def __init__(self, metric, index_params):

        self._bitvector_cutoff = 0
        self._budget = 0
        self.dirty_args = False
        self._dataset = None

        if 'T' in index_params:
            os.environ['PARLAY_NUM_THREADS'] = str(min(int(index_params['T']), os.cpu_count()))

        # Mapping from tag to tag cluster
        self.filter_mapping = {}
        print("metric:", metric)

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
        pass

    def csr_filter_to_str_filter(self, csr_filter):
        filters = []
        for i in range(len(csr_filter.indptr) - 1):
            row_filter = []
            for filter_num in csr_filter.indices[csr_filter.indptr[i]: csr_filter.indptr[i + 1]]:
                row_filter.append(filter_num)
            filters.append(row_filter)
        return filters
    
    def fit(self, dataset):
        self._dataset = dataset
        start = time.time()
        ds = DATASETS[dataset]()
        self.dtype = self.translate_dtype(ds.dtype)
        historical_filters = pickle.load(open("yfcc10M-filters.pkl", "rb"))
        print(historical_filters.shape)

        # there's almost certainly a way to do this in less than 0.1s, which costs us ~200 QPS
        rows, cols = historical_filters.nonzero()
        filter_dict = defaultdict(list)

        for row, col in zip(rows, cols):
            if row <= 25000:
                filter_dict[row].append(col)
            # filter_dict[row].append(col)
        # md = scipy.sparse.csc_matrix(ds.get_dataset_metadata())
        # for comp in range(len(md.indptr) - 1):
        #     sub_indices = md.indices[md.indptr[comp]: md.indptr[comp + 1]]
        #     if len(sub_indices) <= 500:
        #         continue
        #     filter_dict[comp].append(comp)
        # for i in range(historical_filters.shape[1]):
        #     filter_dict[i].append(i)

        # filters = [wp.QueryFilter(*filters[i]) for i in filters.keys()]
        # historical_filters_list = [None] * len(filter_dict.keys())
        # for i in filter_dict.keys():
        #     historical_filters_list[i] = hnswlib.QueryFilter(*filter_dict[i])
        historical_filters_list = []
        for i in filter_dict.keys():
            historical_filters_list.append(hnswlib.QueryFilter(*filter_dict[i]))

        filters_list_first_half = historical_filters_list[:int(len(historical_filters_list) / 2)]
        filters_list_second_half = historical_filters_list[int(len(historical_filters_list) / 2):]

        if hasattr(self, 'index'):
            print("Index already exists, skipping fit")
            return

        self.index = hnswlib.HierarchicalIndex(ds.get_dataset_fn(), os.path.join(ds.basedir, ds.ds_metadata_fn), historical_filters_list, 10000000, 192, 32, 40, 30000000, 1000, 100000, True, 32)
        # self.index = hnswlib.HierarchicalIndex(ds.get_dataset_fn(), os.path.join(ds.basedir, ds.ds_metadata_fn), historical_filters_list, 0,  10000000, 192, 4, 40, 32, 10000000, 0.2)
        # self.index = hnswlib.HierarchicalIndex(ds.get_dataset_fn(), os.path.join(ds.basedir, ds.ds_metadata_fn), historical_filters_list, 100000, 100000, 64, 64, 40, 32, 10000, 0.2)
        # self.index.update_index(filters_list_second_half)
        print("Index initialized")
        # self.index.print_stats()
        print(f"Index fit in {time.time() - start} seconds")

        try:
            pickle.dump(self.index, open("yfcc10M-index.pkl", "wb"))
        except Exception as e:
            print("can't dump index", e)

    def load_index(self, dataset):
        self._dataset = dataset
        start = time.time()
        ds = DATASETS[dataset]()
        self.dtype = self.translate_dtype(ds.dtype)
        historical_filters = pickle.load(open("yfcc10M-filters.pkl", "rb"))
        print(historical_filters.shape)

        # there's almost certainly a way to do this in less than 0.1s, which costs us ~200 QPS
        rows, cols = historical_filters.nonzero()
        filter_dict = defaultdict(list)

        for row, col in zip(rows, cols):
            if row <= 25000:
                filter_dict[row].append(col)
            # filter_dict[row].append(col)
        # md = scipy.sparse.csc_matrix(ds.get_dataset_metadata())
        # for comp in range(len(md.indptr) - 1):
        #     sub_indices = md.indices[md.indptr[comp]: md.indptr[comp + 1]]
        #     if len(sub_indices) <= 5000:
        #         continue
        #     filter_dict[comp].append(comp)
        # for i in range(historical_filters.shape[1]):
        #     filter_dict[i].append(i)

        # filters = [wp.QueryFilter(*filters[i]) for i in filters.keys()]
        # historical_filters_list = [None] * len(filter_dict.keys())
        # for i in filter_dict.keys():
        #     historical_filters_list[i] = hnswlib.QueryFilter(*filter_dict[i])
        historical_filters_list = []
        for i in filter_dict.keys():
            historical_filters_list.append(hnswlib.QueryFilter(*filter_dict[i]))

        filters_list_first_half = historical_filters_list[:int(len(historical_filters_list) / 2)]
        filters_list_second_half = historical_filters_list[int(len(historical_filters_list) / 2):]

        if hasattr(self, 'index'):
            print("Index already exists, skipping fit")
            return

        self.index = hnswlib.HierarchicalIndex(ds.get_dataset_fn(), os.path.join(ds.basedir, ds.ds_metadata_fn), historical_filters_list, 10000000, 192, 32, 40, 30000000, 1000, 100000, True, 32)
        # self.index = hnswlib.HierarchicalIndex(ds.get_dataset_fn(), os.path.join(ds.basedir, ds.ds_metadata_fn), historical_filters_list, 0,  10000000, 192, 4, 40, 32, 10000000, 0.2)
        # self.index = hnswlib.HierarchicalIndex(ds.get_dataset_fn(), os.path.join(ds.basedir, ds.ds_metadata_fn), historical_filters_list, 100000, 100000, 64, 64, 40, 32, 10000, 0.2)
        # self.index.update_index(filters_list_second_half)
        print("Index initialized")
        # self.index.print_stats()
        print(f"Index fit in {time.time() - start} seconds")

        try:
            pickle.dump(self.index, open("yfcc10M-index.pkl", "wb"))
        except Exception as e:
            print("can't dump index", e)

    def set_query_arguments(self, query_args):
        self.dirty_args = True
        self._query_args = query_args
        self._bitvector_cutoff = query_args['bitvector_cutoff']
        self._budget = query_args['budget']
    
    def filtered_query(self, X, filter, k):
        # if self.dirty_args:
        #     self.index_fit_helper()
        start = time.time()

        if k != 10:
            self.set_beamsearch_params(k)

        # there's almost certainly a way to do this in less than 0.1s, which costs us ~200 QPS
        rows, cols = filter.nonzero()
        filter_dict = defaultdict(list)

        for row, col in zip(rows, cols):
            filter_dict[row].append(col)

        # filters = [wp.QueryFilter(*filters[i]) for i in filters.keys()]
        filters = [None] * len(filter_dict.keys())
        for i in filter_dict.keys():
            filters[i] = hnswlib.QueryFilter(*filter_dict[i])

        print(f"Filter construction took {time.time() - start} seconds")
        search_start = time.time()
        nq = X.shape[0]
        self.res = self.index.batch_filter_search(X, filters, nq, k, 32, 1)
        print(self.res.shape)
        print(f"Search took {time.time() - search_start} seconds")

    def get_results(self):
        # print(self.res.shape)
        # print(self.query_dists.shape)
        # print(self.res[:10, :10])
        # print(self.query_dists[:10, :10])
        return np.array(self.res)
    
    def __str__(self):
        return f"MyMethod"
    
    def index_name(self):
        return f"my_method"


# if __name__ == "__main__":
#     index_params = {
#         "cluster_size": 5000, 
#         "T": 8,
#         "cutoff": 10000,
#         "max_iter": 10,
#         "weight_classes": [100000, 400000],
#         "build_params": [{"max_degree": 8,
#                                 "limit": 200,
#                                 "alpha": 1.175},
#                               {"max_degree": 10,
#                                "limit": 200,
#                                "alpha": 1.175},
#                               {"max_degree": 12,
#                                "limit": 200,
#                                "alpha": 1.175}],
#         "bitvector_cutoff": 10000
#     }
#     par = SmartPartition("euclidean", index_params)
#     par.fit("yfcc-10M")
