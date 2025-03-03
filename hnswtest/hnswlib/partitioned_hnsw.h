#pragma once

#include "hnswlib.h"
#include "filters.h"
#include <cmath>
#include <memory>
#include <thread>
#include <atomic>
#include <random>
#include <stdlib.h>
#include <assert.h>
#include <unordered_set>
#include <list>
#include <unordered_map>
#include <bitset>
#include <memory>
#include <queue>
#include <math.h>
#include <iostream>
#include "roaring.hh"
#include "roaring.c"

#define XXH_INLINE_ALL
#include "xxhash.h"


namespace hnswlib {

// Custom hash functor for std::pair<int, int>
struct pair_hash {
    std::size_t operator()(const std::pair<size_t, size_t>& p) const {
        // Combine the hash of the two ints.
        // This is one simple way to do it.
        auto h1 = std::hash<size_t>{}(p.first);
        auto h2 = std::hash<size_t>{}(p.second);
        // The bitwise combination; you can tweak this as needed.
        return h1 ^ (h2 << 1);
    }
};

struct PartitionedIndexParams {
  // dataset parameters
  size_t dataset_size;
  size_t dim;

  // HNSW parameters
  size_t M = 16;
  size_t ef_construction = 200;

  // Partitioned index parameters
  size_t index_vector_budget;
  size_t bitvector_cutoff = 500;

  // Historical workload parameters
  size_t historical_workload_window_size = 100000;

  // Heterogeneous indexing flag
  bool enable_heterogeneous_indexing = true;

  // Hetereogeneous search flag
  bool enable_heterogeneous_search = true;

  // Query correlation constant
  float query_correlation_constant = 0.5;

  size_t num_threads = 8;
};

class PartitionedIndexCounters{
  public:
    // predicate stats (for finding partitions)
    float predicate_construction_time = 0;
    float predicate_comp_time = 0;

    // oracle searches
    size_t oracle_searches = 0;
    float oracle_search_time = 0;

    // (non-root) upward searches
    size_t upward_searches = 0;
    float upward_search_time = 0;
    float upward_search_selectivity = 0;

    // root searches
    size_t root_searches = 0;
    float root_search_time = 0;
    float root_search_selectivity = 0;

    // bruteforce searches
    size_t bruteforce_searches = 0;
    float bruteforce_search_time = 0;
    size_t bruteforce_distance_comps = 0;

    void add_oracle_search(float search_time) {
        oracle_searches++;
        oracle_search_time += search_time;
    }

    void add_upward_search(float search_time, float search_selectivity) {
        upward_searches++;
        upward_search_time += search_time;
        upward_search_selectivity += search_selectivity;
    }

    void add_root_search(float search_time, float search_selectivity) {
        root_searches++;
        root_search_time += search_time;
        root_search_selectivity += search_selectivity;
    }

    void add_bruteforce_search(float search_time, size_t distance_comps) {
        bruteforce_searches++;
        bruteforce_search_time += search_time;
        bruteforce_distance_comps += distance_comps;
    }

    void print_stats() {
        std::cout << "Predicate construction time: " << predicate_construction_time << std::endl << std::flush;
        std::cout << "Predicate comparison time: " << predicate_comp_time << std::endl << std::flush;
        std::cout << "------------------------------------------------------" << std::endl << std::flush;
        std::cout << "Oracle searches: " << oracle_searches << std::endl << std::flush;
        std::cout << "Oracle search time: " << oracle_search_time << std::endl << std::flush;
        std::cout << "------------------------------------------------------" << std::endl << std::flush;
        if (upward_searches > 0) {
            std::cout << "Non-root upward searches: " << upward_searches << std::endl << std::flush;
            std::cout << "Non-root upward search time: " << upward_search_time << std::endl << std::flush;
            std::cout << "Non-root upward search average selectivity: " << upward_search_selectivity / upward_searches << std::endl << std::flush;
            std::cout << "------------------------------------------------------" << std::endl << std::flush;
        }
        if (root_searches > 0) {
            std::cout << "Root searches: " << root_searches << std::endl << std::flush;
            std::cout << "Root search time: " << root_search_time << std::endl << std::flush;
            std::cout << "Root average selectivity: " << root_search_selectivity / root_searches << std::endl << std::flush;
            std::cout << "------------------------------------------------------"  << std::endl << std::flush;
        }
        std::cout << "Bruteforce searches: " << bruteforce_searches << std::endl << std::flush;
        std::cout << "Bruteforce search time: " << bruteforce_search_time << std::endl << std::flush;
        std::cout << "Bruteforce distance comps: " << bruteforce_distance_comps << std::endl << std::flush;
    }

    void clear_stats() {
        predicate_construction_time = 0;
        predicate_comp_time = 0;
        oracle_searches = 0;
        oracle_search_time = 0;
        upward_searches = 0;
        upward_search_time = 0;
        upward_search_selectivity = 0;
        root_searches = 0;
        root_search_time = 0;
        root_search_selectivity = 0;
        bruteforce_searches = 0;
        bruteforce_search_time = 0;
        bruteforce_distance_comps = 0;
    }
};


size_t downscaled_M(size_t cardinality, size_t root_cardinality, size_t M) {
    size_t min_M = 4;
    size_t new_M = std::max(min_M, static_cast<size_t>(std::round((log10(cardinality) / log10(root_cardinality) * M) / 4.0f) * 4.0f));
    return new_M;
}

size_t downscaled_ef_search(size_t cardinality, size_t root_cardinality, size_t ef_search, size_t k) {
    size_t new_ef_search = std::max(k, static_cast<size_t>(log10(cardinality) / log10(root_cardinality) * ef_search));
    return new_ef_search;
}

template<typename dist_t, typename data_t>
class PartitionedHNSWNode{
  public:
    SpaceInterface<dist_t>* _space;
    HierarchicalNSW<dist_t>* _hnsw;
    Predicate _predicate;
    std::vector<PartitionedHNSWNode<dist_t, data_t>*> _children;
    int _id;

    PartitionedHNSWNode(
      Predicate predicate,
      data_t* data,
      SpaceInterface<dist_t>* space,
      PartitionedIndexParams index_params)
    : _space(space), _predicate(predicate) {
        int new_ef = index_params.ef_construction;
        int new_M = index_params.M;
        if (index_params.enable_heterogeneous_indexing) {
            new_M = downscaled_M(predicate.cardinality(), index_params.dataset_size, index_params.M);
        }
        // std::cout << "new M:" << new_M << std::endl;
        _hnsw = new HierarchicalNSW<dist_t>(_space, predicate.cardinality(), new_M, new_ef);
        std::vector<uint32_t> tmp_vec = predicate.matching_points();
        ParallelFor(0, tmp_vec.size(), index_params.num_threads, [&](size_t row, size_t threadId) {
            _hnsw->addPoint((void*)(data + index_params.dim * tmp_vec[row]), tmp_vec[row]);
        });
    }

    void AddPoint(data_t* data, int id) {
        _hnsw->addPoint((void*)(data), id);
    }
};

template<typename dist_t, typename data_t>
class PartitionedHNSW{
  public:
    data_t* _data;
    SpaceInterface<dist_t>* _space;
    DatasetFilters* _dataset_filters;
    PartitionedIndexParams _index_params;
    std::vector<QueryFilter> _historical_workload;
    
    PartitionedHNSWNode<dist_t, data_t>* _root = nullptr;

    std::unordered_map<QueryFilter, PartitionedHNSWNode<dist_t, data_t>*, QueryFilterHash> _node_map; 
    std::vector<PartitionedHNSWNode<dist_t, data_t>*> _nodes;

    PartitionedIndexCounters _index_counters;

    // For cost model
    size_t _ef = 10;
    size_t _k = 10;

    bool has_root = false;

    PartitionedHNSW(
      data_t* data,
      SpaceInterface<dist_t>* space,
      DatasetFilters* dataset_filters,
      PartitionedIndexParams index_params,
      const std::vector<QueryFilter>& historical_workload) : _data(data), _space(space), _dataset_filters(dataset_filters), _index_params(index_params) {
        std::cout << "Dataset size: " << index_params.dataset_size << std::endl << std::flush;
        std::cout << "dimension: " << index_params.dim << std::endl << std::flush;
        std::cout << "M: " << index_params.M << std::endl << std::flush;
        std::cout << "ef_construction: " << index_params.ef_construction << std::endl << std::flush;
        std::cout << "Index vector budget: " << index_params.index_vector_budget << std::endl << std::flush;
        std::cout << "Bitvector cutoff: " << index_params.bitvector_cutoff << std::endl << std::flush;
        std::cout << "Num threads: " << index_params.num_threads << std::endl << std::flush;
        std::cout << "------------------------------------------------------" << std::endl << std::flush;

        updateIndexWorkload(historical_workload);
        fitIndex();
    }

    void updateIndexWorkload(const std::vector<QueryFilter>& new_historical_workload) {
        // Prune historical workload vector according to rolling window size
        _historical_workload.insert(_historical_workload.end(),new_historical_workload.begin(), new_historical_workload.end());
        if (_historical_workload.size() > _index_params.historical_workload_window_size) {
            _historical_workload = std::vector<QueryFilter>(_historical_workload.end() - _index_params.historical_workload_window_size, _historical_workload.end());
        }
    }

    void updateData(
        data_t* _new_data,
        DatasetFilters* _new_dataset_filters,
        size_t num_threads
    ) {
        ParallelFor(0, _nodes.size() + 1, _index_params.num_threads, [&](size_t i, size_t threadId) {
            if (i == _nodes.size()) {
                auto match_bv = _new_dataset_filters->query_matches(_root->_predicate._query_filter);
                // _root->_hnsw->resizeIndex(match_bv->cardinality() + _root->_hnsw->cur_element_count);
                std::vector<uint32_t> tmp_vec;
                for (roaring::Roaring::const_iterator j = match_bv->begin(); j != match_bv->end(); j++) {
                    tmp_vec.push_back(*j);
                }
                ParallelFor(0, tmp_vec.size(), num_threads, [&](size_t row, size_t threadId) {
                    _root->_hnsw->addPoint((void*)(_new_data + _index_params.dim * tmp_vec[row]), _index_params.dataset_size + tmp_vec[row]);
                });
                // std::cout << "inserted points:" << std::endl;
            } else {
                auto match_bv = _new_dataset_filters->query_matches(_nodes[i]->_predicate._query_filter);
                // _nodes[i]->_hnsw->resizeIndex(match_bv->cardinality() + _nodes[i]->_hnsw->cur_element_count);
                std::vector<uint32_t> tmp_vec;
                for (roaring::Roaring::const_iterator j = match_bv->begin(); j != match_bv->end(); j++) {
                    tmp_vec.push_back(*j);
                }
                ParallelFor(0, tmp_vec.size(), num_threads, [&](size_t row, size_t threadId) {
                    _nodes[i]->_hnsw->addPoint((void*)(_new_data + _index_params.dim * tmp_vec[row]), _index_params.dataset_size + tmp_vec[row]);
                });
            }
        });
    }

    void fitIndex() {
        // create predicate objects
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<std::pair<Predicate, size_t>> hw_preds = tally_historical_workloads();
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Time to setup historical query predicates: " << std::chrono::duration<double>(end - start).count() << std::endl << std::flush;

        // Find partitions to construct.
        start = std::chrono::high_resolution_clock::now();
        std::vector<int> selected_partitions = find_partitions(hw_preds);
        end = std::chrono::high_resolution_clock::now();
        std::cout << "Time to select partitions: " << std::chrono::duration<double>(end - start).count() << std::endl << std::flush;

        // Construct partitions
        start = std::chrono::high_resolution_clock::now();
        construct_partitions(hw_preds, selected_partitions);
        end = std::chrono::high_resolution_clock::now();
        std::cout << "Constructed all partitions. time: " << std::chrono::duration<double>(end - start).count() << std::endl << std::flush;

        // Create Hasse diagram over partitions
        start = std::chrono::high_resolution_clock::now();
        create_hasse_diagram();
        end = std::chrono::high_resolution_clock::now();
        std::cout << "Created hasse diagram over partitions. time: " << std::chrono::duration<double>(end - start).count() << std::endl << std::flush;
    }

    std::priority_queue<std::pair<dist_t, labeltype>>
    searchKnnBf(const void *query_data, size_t k, roaring::Roaring* query_bitset, size_t query_cardinality) {
        auto start = std::chrono::high_resolution_clock::now();
        roaring::Roaring& deref = *query_bitset;
        std::priority_queue<std::pair<dist_t, labeltype>> max_priority_queue;
        for (roaring::Roaring::const_iterator i = deref.begin(); i != deref.end(); i++) {
            dist_t dist = _root->_hnsw->fstdistfunc_(query_data, (_data + _index_params.dim * *i), _root->_hnsw->dist_func_param_);
            if (max_priority_queue.size() < k) {
                max_priority_queue.push(std::make_pair(dist, *i));
            } else if (dist < max_priority_queue.top().first) {
                max_priority_queue.pop();
                max_priority_queue.push(std::make_pair(dist, *i));
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        _index_counters.add_bruteforce_search(std::chrono::duration<double>(end - start).count(), query_cardinality);
        return max_priority_queue;
    }

    void setEf(size_t ef) {
        _ef = ef;
        _root->_hnsw->setEf(ef);
        for (auto &partition : _nodes) {
            if (_index_params.enable_heterogeneous_search) {
                partition->_hnsw->setEf(downscaled_ef_search(partition->_predicate.cardinality(), _index_params.dataset_size, ef, 10));
            } else {
                partition->_hnsw->setEf(ef);
            }
        }
    }

    std::priority_queue<std::pair<dist_t, labeltype>>
    searchKnn(const void *query_data, size_t k, Predicate query_predicate) {
        _k = k;
        // Bruteforce results for selective queries
        if (query_predicate.cardinality() <= _index_params.bitvector_cutoff) {
            return searchKnnBf(query_data, k, query_predicate._bitvector, query_predicate.cardinality());
        }

        auto start = std::chrono::high_resolution_clock::now();
        BitMapFilter QueryBitset(query_predicate._bitvector);
        auto end = std::chrono::high_resolution_clock::now();
        _index_counters.predicate_construction_time += std::chrono::duration<double>(end - start).count();

        PartitionedHNSWNode<dist_t, data_t>* best_partition = _root;
        size_t best_size = _root->_predicate.cardinality();

        // See if oracle partition is constructed
        start = std::chrono::high_resolution_clock::now();
        if (_node_map.find(query_predicate._query_filter) != _node_map.end()) {
            best_size = _node_map[query_predicate._query_filter]->_predicate.cardinality();
            best_partition = _node_map[query_predicate._query_filter];
        } else {
            // BFS in hasse diagram
            std::queue<PartitionedHNSWNode<dist_t, data_t>*> bfs_queue;
            std::unordered_set<int> visited;
            bfs_queue.push(_root);
            while (!bfs_queue.empty()) {
                PartitionedHNSWNode<dist_t, data_t>* cur = bfs_queue.front();
                bfs_queue.pop();
                visited.insert(cur->_id);
    
                if (cur->_predicate.cardinality() < best_size) {
                    best_size = cur->_predicate.cardinality();
                    best_partition = cur;
                }
    
                for (int i = 0; i < cur->_children.size(); i++) {
                    if (visited.find(cur->_children[i]->_id) == visited.end() && cur->_children[i]->_predicate.cardinality() >= query_predicate.cardinality() && query_predicate.is_logical_subset(cur->_children[i]->_predicate)) {
                        bfs_queue.push(cur->_children[i]);
                    }
                }
            }
        }
        
        end = std::chrono::high_resolution_clock::now();
        _index_counters.predicate_comp_time += std::chrono::duration<double>(end - start).count();

        // Check whether Bruteforce search is cheaper
        float cur_upward_search_cost = upward_search_cost(best_size, query_predicate.cardinality());
        float cur_bf_search_cost = bf_search_cost(query_predicate.cardinality());
        if (cur_upward_search_cost >= cur_bf_search_cost) {
            return searchKnnBf(query_data, k, query_predicate._bitvector, query_predicate.cardinality());
        }
        start = std::chrono::high_resolution_clock::now();
        auto res = best_partition->_hnsw->searchKnn(query_data, k, &QueryBitset);
        end = std::chrono::high_resolution_clock::now();

        // Update counter
        if (best_size == query_predicate.cardinality()) {
            _index_counters.add_oracle_search(std::chrono::duration<double>(end - start).count());
        } else if (best_size != _index_params.dataset_size) {
            _index_counters.add_upward_search(std::chrono::duration<double>(end - start).count(), (float) query_predicate.cardinality() / best_size);
        } else {
            _index_counters.add_root_search(std::chrono::duration<double>(end - start).count(), (float) query_predicate.cardinality() / (float) _index_params.dataset_size);
        }
        
        return res;
    }

    void printTally() {
        _index_counters.print_stats();
        _index_counters.clear_stats();

        // Calculate total distance hops
        long total_hops = _root->_hnsw->metric_distance_computations;
        _root->_hnsw->metric_distance_computations = 0;
        for (auto &i : _nodes) {
            total_hops += i->_hnsw->metric_distance_computations;
            i->_hnsw->metric_distance_computations = 0;
        }
        std::cout << "HNSW distance calcs: " << total_hops << std::endl << std::flush;
    }

  private:
    std::vector<std::pair<Predicate, size_t>> tally_historical_workloads() {
        std::vector<std::pair<QueryFilter, size_t>> tmp_vec = tally_query_filters(_historical_workload, _dataset_filters, _index_params.bitvector_cutoff);
        std::vector<std::pair<Predicate, size_t>> historical_workload_preds(tmp_vec.size());
        ParallelFor(0, tmp_vec.size(), _index_params.num_threads, [&](size_t i, size_t threadId) {
            historical_workload_preds[i] = std::make_pair(Predicate(_dataset_filters, tmp_vec[i].first), tmp_vec[i].second);
        }); 
        std::sort(historical_workload_preds.begin(), historical_workload_preds.end(), [](const auto &a, const auto &b)
        {
            return a.first.cardinality() < b.first.cardinality(); 
        });
        std::cout << "candidate partitions: " << historical_workload_preds.size() << std::endl << std::flush;
        return historical_workload_preds;
    }

    std::vector<int> find_partitions(const std::vector<std::pair<Predicate, size_t>>& hw_preds) {
        // Find edges
        std::unordered_map<int, std::unordered_set<int>> parent_set;
        std::unordered_map<int, std::unordered_set<int>> child_set;
        int num_edges = 0;

        // for (int i = 0; i < hw_preds.size(); i++) {
        //     std::cout << i << " pred size:" << hw_preds[i].first.cardinality();
        // }

        std::vector <std::pair<size_t, size_t>> edges[_index_params.num_threads];
        ParallelFor(0, hw_preds.size(), _index_params.num_threads, [&](size_t i, size_t threadId) {
            for (size_t j = i; j < hw_preds.size(); j++) {
                if (hw_preds[i].first.is_logical_subset(hw_preds[j].first)) {
                    edges[threadId].push_back(std::make_pair(i, j));
                }
            }
        });

        for (size_t i = 0; i < _index_params.num_threads; i++) {
            for (size_t j = 0; j < edges[i].size(); j++) {
                num_edges++;
                size_t child = edges[i][j].first;
                size_t parent = edges[i][j].second;
                if (parent_set.find(child) != parent_set.end()) {
                    parent_set[child].insert(parent);
                } else {
                    parent_set[child] = std::unordered_set<int>{parent};
                }
                if (child_set.find(parent) != child_set.end()) {
                    child_set[parent].insert(child);
                } else {
                    child_set[parent] = std::unordered_set<int>{child};
                }
            }
        }

        // Submodular optimization
        size_t total_vecs = 0;

        std::vector<float> best_costs;  // Current best cost tracker
        std::vector<bool> dirty;  // Dirty tracker
        for (size_t i = 0; i < hw_preds.size(); i++) {
            best_costs.push_back(std::min(bf_search_cost(hw_preds[i].first.cardinality()), root_search_cost(hw_preds[i].first.cardinality())));
            dirty.push_back(false);
        }

        // Compute initial round of marginal benefits
        std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, std::less<std::pair<float, int>>> max_queue;
        for (size_t i = 0; i < hw_preds.size(); i++) {
            float ratio_sum = 0;
            // Cost of applicable children
            for (const auto& child: child_set[i]) {
                ratio_sum += std::max(0.0f, best_costs[child] - upward_search_cost(hw_preds[child].first.cardinality(), hw_preds[i].first.cardinality())) * hw_preds[child].second;
            }
            max_queue.push(std::make_pair(ratio_sum / scaled_partition_size(hw_preds[i].first.cardinality()), i));
        }

        std::cout << "edges:" << num_edges << std::endl << std::flush;
        std::cout << "Initialized submodular optimizer." << std::endl << std::flush;

        std::unordered_set<int> selected;
        size_t total_budget = (size_t) (((float)_index_params.index_vector_budget / (float)_index_params.dataset_size) * scaled_partition_size(_index_params.dataset_size));
        std::cout << "total vecs:" << total_vecs;
        std::cout << "total budget:" << total_budget;

        while (total_vecs < total_budget && selected.size() < hw_preds.size()) {
            std::pair<float, int> top = max_queue.top(); 
            max_queue.pop();
            int node = top.second;

            // Already selected
            if (selected.find(node) != selected.end()) {
                continue;
            }

            // popped benefit of current node is inaccurate (due to a parent or child being selected); recompute it and add it back to the pqueue.
            if (dirty[node]) {
                float ratio_sum = 0;
                // Cost of applicable children
                for (const auto& child: child_set[node]) {
                    ratio_sum += std::max(0.0f, best_costs[child] - upward_search_cost(hw_preds[child].first.cardinality(), hw_preds[node].first.cardinality())) * hw_preds[child].second;
                }
                max_queue.push(std::make_pair(ratio_sum / scaled_partition_size(hw_preds[node].first.cardinality()), node));
                dirty[node] = false;
                continue;
            }

            // Select the node and mark the costs of all its children and parents as dirty.
            for (const auto& child: child_set[node]) {
                if (selected.find(child) == selected.end()) {
                    dirty[child] = true;
                }
            }
            for (const auto& parent: parent_set[node]) {
                if (selected.find(parent) == selected.end()) {
                    dirty[parent] = true;
                }
            }

            // Update the scores of all queries potentially handled at this node.
            for (const auto& child: child_set[node]) {
                best_costs[child] = std::min(upward_search_cost(hw_preds[child].first.cardinality(), hw_preds[node].first.cardinality()), best_costs[child]); 
            }

            selected.insert(node);
            total_vecs += scaled_partition_size(hw_preds[node].first.cardinality()); 
        }

        std::cout << "Number of partitions: " << selected.size() << std::endl << std::flush;

        // std::ofstream outFile1("/home/zl20/yfcc-partitons-constructed.txt");
        // std::unordered_set<std::pair<size_t, size_t>, pair_hash> printed;
        // for (size_t i = 0; i < hw_preds.size(); i++) {
        //     if (selected.find(i) != selected.end()) {
        //         if (printed.find(std::make_pair(hw_preds[i].first.cardinality(), hw_preds[i].second)) == printed.end()) {
        //             outFile1 << hw_preds[i].first.cardinality() << " " << hw_preds[i].second << "\n";
        //             printed.insert(std::make_pair(hw_preds[i].first.cardinality(), hw_preds[i].second));
        //         }
                
        //     }
        // }
        // outFile1.close();

        // std::ofstream outFile2("/home/zl20/yfcc-partitons-deleted.txt");
        // std::unordered_set<std::pair<size_t, size_t>, pair_hash> printed2;
        // for (size_t i = 0; i < hw_preds.size(); i++) {
        //     if (selected.find(i) == selected.end()) {
        //         if (printed2.find(std::make_pair(hw_preds[i].first.cardinality(), hw_preds[i].second)) == printed2.end()) {
        //             outFile2 << hw_preds[i].first.cardinality() << " " << hw_preds[i].second << "\n";
        //             printed2.insert(std::make_pair(hw_preds[i].first.cardinality(), hw_preds[i].second));
        //         }
                
        //     }
        // }
        // outFile2.close();

        // Log stuff
        num_edges = 0;
        std::vector<int> selected_partitions;
        for (size_t i = 0; i < hw_preds.size(); i++) {
            if (selected.find(i) != selected.end()) {
                selected_partitions.push_back(i);
                for (const auto& child: child_set[i]) {
                    if (selected.find(child) != selected.end()) {
                        num_edges++;
                    }
                }
            }
        }
        std::cout << "Number of edges: " << num_edges << std::endl << std::flush;

        return selected_partitions;
    }

    void construct_partitions(std::vector<std::pair<Predicate, size_t>> hw_preds, std::vector<int> selected_partitions) {
        // Delete outdated partitions
        std::unordered_set<QueryFilter, QueryFilterHash> partitions_to_construct;
        for (int idx : selected_partitions) {
            partitions_to_construct.insert(hw_preds[idx].first._query_filter);
        }
        std::vector<QueryFilter> to_delete;
        for (auto &it: _node_map) {
            if (partitions_to_construct.find(it.first) == partitions_to_construct.end()) {
                to_delete.push_back(it.first);
            }
        }
        for (auto&& key : to_delete) {
            delete _node_map[key];
            _node_map.erase(key);
        }
        std::cout << "Deleted partitions: " << to_delete.size() << std::endl << std::flush;

        // Find new partitions to construct
        std::vector<int> new_selected_partitions;
        size_t new_vec_count = 0;
        
        for (int idx : selected_partitions) {
            if (_node_map.find(hw_preds[idx].first._query_filter) == _node_map.end()) {
                new_selected_partitions.push_back(idx);
                new_vec_count += hw_preds[idx].first.cardinality();
            }
        }
        std::cout << "New partitions to construct: " << new_selected_partitions.size() << std::endl << std::flush;
        std::cout << "New partitions vectors: " << new_vec_count << std::endl << std::flush;

        if (!has_root) {
            has_root = true;
            // Construct partitions
            ParallelFor(0, new_selected_partitions.size() + 1, _index_params.num_threads, [&](size_t i, size_t threadId) {
                if (i == new_selected_partitions.size() && _root == nullptr) {
                    _root = new PartitionedHNSWNode<dist_t, data_t>(Predicate(_index_params.dataset_size), _data, _space, _index_params);
                } else {
                   _node_map[hw_preds[selected_partitions[i]].first._query_filter] = new PartitionedHNSWNode<dist_t, data_t>(hw_preds[new_selected_partitions[i]].first, _data, _space, _index_params);
                }
            });
        } else {
            // Construct partitions
            ParallelFor(0, new_selected_partitions.size(), _index_params.num_threads, [&](size_t i, size_t threadId) {
                   _node_map[hw_preds[selected_partitions[i]].first._query_filter] = new PartitionedHNSWNode<dist_t, data_t>(hw_preds[new_selected_partitions[i]].first, _data, _space, _index_params);
            });
        }
        
        std::cout << "Updated partitions." << std::endl << std::flush;

        // Create nodes vector
        _nodes.clear();
        for (auto &it: _node_map) {
            _nodes.push_back(it.second);
        }
        std::sort(_nodes.begin(), _nodes.end(), [](auto &a, auto &b)
        {
            return a->_predicate.cardinality() < b->_predicate.cardinality(); 
        });
        std::cout << "Done." << std::endl << std::flush;

        // Assign IDs
        for (size_t i = 0; i < _nodes.size(); i++) {
            _nodes[i]->_id = i;
        }
        _root->_id = -1;

        // Log sizes
        // for (size_t i = 0; i < _nodes.size(); i++) {
        //     std::cout << "Points:" << _nodes[i]->_predicate.cardinality() << std::endl;
        //     std::cout << "Downscaled M:" << downscaled_M(_nodes[i]->_predicate.cardinality(), _index_params.dataset_size, _index_params.M) << std::endl;
        //     std::cout << "Index size:" << _nodes[i]->_hnsw->indexFileSize() << std::endl;
        //     std::cout << "Predicted size:" << scaled_partition_size(_nodes[i]->_predicate.cardinality()) << std::endl;
        // }
    }

    void create_hasse_diagram() {
        // Find edges in constructed partitions
        std::unordered_map<int, std::unordered_set<int>> parent_set;
        std::unordered_map<int, std::unordered_set<int>> child_set;
        int num_edges = 0;

        std::vector <std::pair<size_t, size_t>> edges[_index_params.num_threads];
        ParallelFor(0, _nodes.size(), _index_params.num_threads, [&](size_t i, size_t threadId) {
            for (size_t j = i + 1; j < _nodes.size(); j++) {
                if (_nodes[i]->_predicate.is_logical_subset(_nodes[j]->_predicate)) {
                    edges[threadId].push_back(std::make_pair(i, j));
                }
            }
        });

        for (size_t i = 0; i < _index_params.num_threads; i++) {
            for (size_t j = 0; j < edges[i].size(); j++) {
                num_edges++;
                if (parent_set.find(i) != parent_set.end()) {
                    parent_set[i].insert(j);
                } else {
                    parent_set[i] = std::unordered_set<int>(j);
                }
                if (child_set.find(j) != child_set.end()) {
                    child_set[j].insert(i);
                } else {
                    child_set[j] = std::unordered_set<int>(i);
                }
            }
        }

        // Clear children vectors in nodes
        for (size_t i = 0; i < _nodes.size(); i++) {
            _nodes[i]->_children.clear();
        }
        _root->_children.clear();

        // Fill children vectors in nodes.
        // Note: _nodes is already sorted w.r.t. ascending cardinality.
        size_t hasse_diagram_edges = 0;
        for (size_t i = 0; i < _nodes.size(); i++) {
            if (parent_set.find(i) == parent_set.end()) {
                continue;
            }
            for (size_t j = i + 1; j < _nodes.size(); j++) {
                // j is a parent of i
                if (parent_set[i].find(j) != parent_set[i].end()) {
                    // Check for direct relationship by ensuring no intermediate 
                    for (size_t k = i + 1; k < j; k++) {
                        if (parent_set.find(k) == parent_set.end()) {
                            continue;
                        }
                        if (parent_set[i].find(k) != parent_set[i].end() && parent_set[k].find(j) != parent_set[k].end()) {
                            break;
                        }
                    }
                    
                    // If direct subset relationship, add as a child in the Hasse diagram
                    hasse_diagram_edges++;
                    _nodes[i]->_children.push_back(_nodes[j]);
                }
            }
        }

        // Connect top-level partitions (i.e., those with no parents) with root.
        for (size_t i = 0; i < _nodes.size(); i++) {
            if (parent_set.find(i) == parent_set.end()) {
                _root->_children.push_back(_nodes[i]);
                hasse_diagram_edges++;
            }
        }
        std::cout << "Hasse diagram edges: " << hasse_diagram_edges << std::endl;
    }

    float bf_search_cost(size_t query_cardinality) {
        return query_cardinality * log(_index_params.bitvector_cutoff) / _index_params.bitvector_cutoff;
    }

    float upward_search_cost(size_t parent_cardinality, size_t query_cardinality) {
        float M_ratio = 1.0f;
        float new_ef_ratio = 1.0f;
        // if (_index_params.enable_heterogeneous_indexing) {
        //     M_ratio = (float) (downscaled_M(parent_cardinality, _index_params.dataset_size, _index_params.M) + 12) / (_index_params.M + 12);
        // }
        if (_index_params.enable_heterogeneous_search) {
            float new_ef = downscaled_ef_search(parent_cardinality, _index_params.dataset_size, _ef, _k);
            new_ef_ratio = (_k + ((new_ef - _k) / 2)) / (float) _k;
        } else {
            new_ef_ratio = (_k + ((_ef - _k) / 2)) / (float) _k;
        }
        return log(parent_cardinality) * pow(parent_cardinality / query_cardinality, _index_params.query_correlation_constant) * new_ef_ratio * M_ratio;
    }
    
    float root_search_cost(size_t query_cardinality) {
        float new_ef_ratio = (_k + ((_ef - _k) / 2)) / (float) _k;
        return log(_index_params.dataset_size) * pow(_index_params.dataset_size / query_cardinality, _index_params.query_correlation_constant) * new_ef_ratio;
    }

    size_t partition_size(size_t cardinality, size_t M) {
        // Estimated partition size
        // size_t total_size = 96;  // misc. items
        // total_size += (M * 8 + 12) * cardinality;  // elements
        // total_size += 9 * cardinality;
        return cardinality * (M + 50) / 82;
        // return total_size;
    }

    size_t scaled_partition_size(size_t cardinality) {
        // Compute M: empirical equation from pinecone (https://www.pinecone.io/learn/series/faiss/hnsw/)
        if (_index_params.enable_heterogeneous_indexing) {
            return partition_size(cardinality, downscaled_M(cardinality, _index_params.dataset_size, _index_params.M));
        }
        return partition_size(cardinality, _index_params.M);
    }
};

        
        // Compute relative benefits of all partitions
        // std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, std::greater<std::pair<float, int>>> min_queue;
        // for (size_t i = 0; i < hw_preds.size(); i++) {
        //     float ratio_sum = 0;
        //     for (auto pr : best_partitions[i]) {
        //         std::vector<float> alt_latencies;
        //         alt_latencies.push_back(root_latencies[pr.first]); // root search
        //         alt_latencies.push_back(bf_latencies[pr.first]); // bruteforce search
        //         for (const auto& parent: parent_set[i]) { // parent searches
        //             alt_latencies.push_back(
        //                 upward_search_cost(hw_preds[parent].first.cardinality(), hw_preds[pr.first].first.cardinality()));
        //         }
        //         ratio_sum += pr.second * (*min_element(alt_latencies.begin(), alt_latencies.end()) - upward_search_cost(hw_preds[i].first.cardinality(), hw_preds[pr.first].first.cardinality())) / scaled_partition_size(hw_preds[i].first.cardinality());
        //             // scaled_partition_size(hw_preds[i].first.cardinality());
        //     }
        //     min_queue.push(std::make_pair(ratio_sum, i));
        // }

        // std::cout << "edges:" << num_edges << std::endl << std::flush;
        // std::cout << "Initialized submodular optimizer." << std::endl << std::flush;

        // std::unordered_set<int> deleted;
        // size_t total_budget = (size_t) (((float)_index_params.index_vector_budget / (float)_index_params.dataset_size) * scaled_partition_size(_index_params.dataset_size));
        // std::cout << "total vecs:" << total_vecs;
        // std::cout << "total budget:" << total_budget;
        // while (total_vecs > total_budget) {
        //     std::pair<float, int> top = min_queue.top(); 
        //     min_queue.pop();
        //     int node = top.second;
            
        //     if (deleted.find(node) != deleted.end()) {
        //         continue;
        //     }
            
        //     // popped benefit of current node is inaccurate (due to a parent or child being deleted); recompute it and add it back to the pqueue.
        //     if (dirty[node]) {
        //         float ratio_sum = 0;
        //         for (auto pr : best_partitions[node]) {
        //             std::vector<float> alt_latencies;
        //             alt_latencies.push_back(root_latencies[pr.first]); // root search
        //             alt_latencies.push_back(bf_latencies[pr.first]); // bruteforce search
        //             for (const auto& parent: parent_set[node]) { // parent searches
        //                 alt_latencies.push_back(
        //                     upward_search_cost(hw_preds[parent].first.cardinality(), hw_preds[pr.first].first.cardinality()));
        //             }
        //             ratio_sum += pr.second * (*min_element(alt_latencies.begin(), alt_latencies.end()) - upward_search_cost(hw_preds[node].first.cardinality(), hw_preds[pr.first].first.cardinality())) / scaled_partition_size(hw_preds[node].first.cardinality());
        //             // scaled_partition_size(hw_preds[node].first.cardinality());
        //         }
        //         min_queue.push(std::make_pair(ratio_sum, node));
        //         dirty[node] = false;
        //         continue;
        //     }

        //     // Delete the node and mark the costs of all its children and parents as dirty.
        //     for (const auto& child: child_set[node]) {
        //         if (deleted.find(child) == deleted.end()) {
        //             dirty[child] = true;
        //         }
        //     }
        //     for (const auto& parent: parent_set[node]) {
        //         if (deleted.find(parent) == deleted.end()) {
        //             dirty[parent] = true;
        //         }
        //     }

        //     // Propagate all queries handled at this node to its smallest parent.
        //     int smallest_parent_id = -1;
        //     size_t smallest_parent_cardinality = _index_params.dataset_size;
        //     for (const auto& parent: parent_set[node]) {
        //         if (deleted.find(parent) == deleted.end()) {
        //             if (hw_preds[parent].first.cardinality() < smallest_parent_cardinality) {
        //                 smallest_parent_cardinality = hw_preds[parent].first.cardinality();
        //                 smallest_parent_id = parent;
        //             }
        //         }
        //     }

        //     if (smallest_parent_id != -1) {
        //         best_partitions[smallest_parent_id].insert(
        //             best_partitions[smallest_parent_id].end(), 
        //             best_partitions[node].begin(),
        //             best_partitions[node].end()
        //         );
        //     }
        //     deleted.insert(node);
        //     total_vecs -= scaled_partition_size(hw_preds[node].first.cardinality()); 
        // }

} // namespace hnswlib