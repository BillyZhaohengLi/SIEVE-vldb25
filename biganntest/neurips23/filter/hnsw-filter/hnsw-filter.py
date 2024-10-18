import heapq
import numpy as np
import os
import pickle
import random
import scipy
import shutil
import time
import xxhash
import hnswlib
import networkx as nx
from pympler import asizeof

from collections import defaultdict
from datasketch import MinHash, MinHashLSH, LeanMinHash
from sklearn.cluster import DBSCAN, KMeans
from unionfind import UnionFind

from neurips23.filter.base import BaseFilterANN
from benchmark.datasets import DATASETS
from benchmark.dataset_io import download_accelerated

class HnswFilter(BaseFilterANN):

    def __init__(self, metric, index_params):

        if 'bitvector_cutoff' in index_params:
            self._bitvector_cutoff = index_params['bitvector_cutoff']
        else:
            self._bitvector_cutoff = self._cutoff

        if 'T' in index_params:
            os.environ['PARLAY_NUM_THREADS'] = str(min(int(index_params['T']), os.cpu_count()))

        # Mapping from tag to tag cluster
        self.filter_mapping = {}
        self.point_count = 0
        self.hnsws = {}

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
        pass
        # start = time.time()
        # ds = DATASETS[dataset]()
        # self.ds_array = ds.get_dataset()
        # self.ds_array = self.ds_array[:1000000]
        # md = ds.get_dataset_metadata()[:1000000]

        # md = scipy.sparse.csc_matrix(md)
        # print("nonzero count:", md.count_nonzero())

        # sum_size = 0

        # for comp in range(len(md.indptr) - 1):
        #     #if comp > 1000:
        #     #    break
        #     # Compute qualifying vectors
        #     print("range:", md.indptr[comp], md.indptr[comp + 1])
        #     sub_indices = md.indices[md.indptr[comp]: md.indptr[comp + 1]]
        #     if len(sub_indices) <= 50:
        #         continue
        #     sub_dataset = self.ds_array[sub_indices]

        #     self.hnsws = hnswlib.Index(space='l2', dim=sub_dataset.shape[1])
        #     self.hnsws.init_index(max_elements=sub_dataset.shape[0], ef_construction=200, M=64)
        #     self.hnsws.add_items(data = sub_dataset, ids=np.arange(sub_dataset.shape[0]))

        #     hnsw_bytestring = pickle.dumps(self.hnsws)
        #     sum_size += asizeof.asizeof(hnsw_bytestring)
        #     del self.hnsws
        #     #print(f"index elements: {sub_dataset.shape[0]}")
        #     #print(f"index size: {asizeof.asizeof(hnsw_bytestring)}")

        # # Pickle and compute size
        # #pickle.dump(self.hnsws, open("hnsws.pkl","wb"))
        # # hnsw_bytestring = pickle.dumps(self.hnsws)
        # # print(f"index size: {asizeof.asizeof(hnsw_bytestring)}")
        # print(f"total index size: {sum_size}")

        # print(f"Index fit in {time.time() - start} seconds")

    def load_index(self, dataset):
        pass
        # start = time.time()
        # ds = DATASETS[dataset]()
        # self.ds_array = ds.get_dataset()
        # self.ds_array = self.ds_array[:1000000]
        # md = ds.get_dataset_metadata()[:1000000]

        # md = scipy.sparse.csc_matrix(md)
        # print("nonzero count:", md.count_nonzero())

        # sum_size = 0

        # for comp in range(len(md.indptr) - 1):
        #     #if comp > 1000:
        #     #    break
        #     # Compute qualifying vectors
        #     sub_indices = md.indices[md.indptr[comp]: md.indptr[comp + 1]]
        #     if len(sub_indices) <= 50:
        #         continue
        #     sub_dataset = self.ds_array[sub_indices]

        #     self.hnsws = hnswlib.Index(space='l2', dim=sub_dataset.shape[1])
        #     self.hnsws.init_index(max_elements=sub_dataset.shape[0], ef_construction=200, M=64)
        #     self.hnsws.add_items(data = sub_dataset, ids=np.arange(sub_dataset.shape[0]))

        #     hnsw_bytestring = pickle.dumps(self.hnsws)
        #     sum_size += asizeof.asizeof(hnsw_bytestring)
        #     del self.hnsws
        #     #print(f"index elements: {sub_dataset.shape[0]}")
        #     #print(f"index size: {asizeof.asizeof(hnsw_bytestring)}")

        # # Pickle and compute size
        # #pickle.dump(self.hnsws, open("hnsws.pkl","wb"))
        # #hnsw_bytestring = pickle.dumps(self.hnsws)
        # #print(f"index size: {asizeof.asizeof(hnsw_bytestring)}")
        # print(f"total index size: {sum_size}")

        # print(f"Index fit in {time.time() - start} seconds")
    
    def filtered_query(self, X, filter, k):
        start = time.time()

            # # there's almost certainly a way to do this in less than 0.1s, which costs us ~200 QPS
        pickle.dump(filter, open("yfcc10M-filters.pkl", "wb"))
        rows, cols = filter.nonzero()
        filter_dict = defaultdict(set)

        # Translate filters
        for row, col in zip(rows, cols):
            filter_dict[row].add(col)

        ds = DATASETS["yfcc-10M"]()
        self.ds_array = ds.get_dataset()
        self.ds_array = self.ds_array[:1000000]
        md = ds.get_dataset_metadata()[:1000000]
        md = scipy.sparse.csc_matrix(md)
        print("nonzero count:", md.count_nonzero())

        sum_size = 0

        # Find unique filters
        unique_filters = set()
        for query in filter_dict.keys():
            unique_filters.add(frozenset(filter_dict[query]))
        print("unique filters:", len(unique_filters))

        unique_large_filters = set()
        unique_bitmaps = set()
        unique_large_bitmaps = set()
        for filterset in unique_filters:
            sub_indices = None
            for filt in filterset:
                if sub_indices is None:
                    sub_indices = set(md.indices[md.indptr[filt]: md.indptr[filt + 1]])
                else:
                    sub_indices = sub_indices.intersection(set(md.indices[md.indptr[filt]:md.indptr[filt + 1]]))
            unique_bitmaps.add(frozenset(sub_indices))

            if len(sub_indices) > 0:
                unique_large_bitmaps.add(frozenset(sub_indices))
                unique_large_filters.add(filterset)
        print("unique bitmaps:", len(unique_bitmaps))
        print("unique large filters:", len(unique_large_filters))
        print("unique large bitmaps:", len(unique_large_bitmaps))

        filterset_2filt = set()
        filterset_1filt = set()
        for filterset in unique_large_filters:
            if len(filterset) == 2:
                filterset_2filt.add(filterset)
            else:
                filterset_1filt.add(min(filterset))

        model1_edges = 0
        for filterset in filterset_2filt:
            for filt in filterset:
                if filt in filterset_1filt:
                    model1_edges += 1
        print(model1_edges)

        def dfs_helper(currnode, unique_large_bitmaps, visited, G):
            if currnode in visited:
                return
            visited.add(currnode)
            G.add_node(currnode)
            for candnode in range(currnode + 1, len(unique_large_bitmaps)):
                if unique_large_bitmaps[currnode].issubset(unique_large_bitmaps[candnode]):
                    if candnode not in G:
                        G.add_node(candnode)
                        G.add_edge(currnode, candnode)
                        dfs_helper(candnode, unique_large_bitmaps, visited, G)
                    elif not nx.has_path(G, currnode, candnode):
                        G.add_edge(currnode, candnode)

        # Construct Model 2
        unique_large_bitmaps = list(unique_large_bitmaps)
        unique_large_bitmaps.sort(key=lambda x: len(x))

        G = nx.DiGraph()

        visited = set()
        for i in range(len(unique_large_bitmaps)):
            # G.add_node(i)
            # for j in range(i, len(unique_large_bitmaps)):
            #     if unique_large_bitmaps[i].issubset(unique_large_bitmaps[j]):
            #         G.add_node(j)
            #         G.add_edge(i, j)
            dfs_helper(i, unique_large_bitmaps, visited, G)
                    
        

        
        # for currnode in range(len(unique_large_bitmaps)):
        #     if currnode in G.nodes():
        #         # a bitvector can only be subsumed by its sibling bitvectors.
        #         toremove = set()
        #         for ancestornode in nx.ancestors(G, currnode):
        #             for successornode in G.successors(ancestornode):
        #                 if successornode != currnode and unique_large_bitmaps[currnode].issubset(unique_large_bitmaps[successornode]):
        #                     G.add_edge(currnode, successornode)
        #                     if successornode in G.children(ancestornode):
        #                         toremove.add((ancestornode, successornode))
        #         for ancestornode, successornode in tmp:
        #             G.remove_edge(ancestornode, successornode)
        #     else:
        #         # Check larger bitvectors for containment
        #         for candnode in range(currnode + 1, len(unique_large_bitmaps)):
        #             if unique_large_bitmaps[currnode].issubset(unique_large_bitmaps[candnode]):
        #                 G.add_node(currnode)
        #                 G.add_node(candnode)
        #                 G.add_edge(currnode, candnode)

        print(len(G.edges()))

        

            # if len(sub_indices) <= 50:
            #     continue
            # #print(np.array(sub_indices).dtype)
            # sub_dataset = self.ds_array[np.array(list(sub_indices)).astype(int)]
            # #print("elements:", sub_dataset.shape[0])

            # self.hnsws = hnswlib.Index(space='l2', dim=sub_dataset.shape[1])
            # self.hnsws.init_index(max_elements=sub_dataset.shape[0], ef_construction=200, M=64)
            # self.hnsws.add_items(data = sub_dataset, ids=np.arange(sub_dataset.shape[0]))

            # hnsw_bytestring = pickle.dumps(self.hnsws)
            # sum_size += asizeof.asizeof(hnsw_bytestring)
            # del self.hnsws
            #print(f"index elements: {sub_dataset.shape[0]}")
            #print(f"index size: {asizeof.asizeof(hnsw_bytestring)}")

        # Pickle and compute size
        #pickle.dump(self.hnsws, open("hnsws.pkl","wb"))
        #hnsw_bytestring = pickle.dumps(self.hnsws)
        #print(f"index size: {asizeof.asizeof(hnsw_bytestring)}")
        # print(f"total index size: {sum_size}")

        # # there's almost certainly a way to do this in less than 0.1s, which costs us ~200 QPS
        # rows, cols = filter.nonzero()
        # filter_dict = defaultdict(set)

        # # Translate filters
        # for row, col in zip(rows, cols):
        #     filter_dict[row].add(col)

        # print(f"Filter construction took {time.time() - start} seconds")
        # search_start = time.time()
        # nq = X.shape[0]

        # self.res = []
        # for query in filter_dict.keys():
        #     # Gather results
        #     valid_indexes = None
        #     for filt in filter_dict[query]:
        #         bv = bitvector.BitVector(size = self.point_count)
        #         if valid_indexes is None:
        #             valid_indexes = self.filter_mapping[filt]
        #         else:
        #             valid_indexes = valid_indexes.intersection(self.filter_mapping[filt])
        #         # Use bitmap
        #         filter_function = lambda x: x in valid_indexes
        #     labels, distances = self.index_hnsw_filter.knn_query(X[query], k=k, num_threads=1, filter=filter_function)
        #     self.res.append(labels)

        print(f"Search took {time.time() - search_start} seconds")

    def get_results(self):
        # print(self.res.shape)
        # print(self.query_dists.shape)
        # print(self.res[:10, :10])
        # print(self.query_dists[:10, :10])
        return np.array(self.res)
    
    def __str__(self):
        return f"FilteredVamana"
    
    def index_name(self):
        return f"filtered_vamana"


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
