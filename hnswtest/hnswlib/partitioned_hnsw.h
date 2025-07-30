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

  // ef-search scaling constant
  float ef_search_scaling_constant = 3;

  // Enable multi-partition search
  bool enable_multipartition_search = false;
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

    size_t covering_searches = 0;
    size_t covering_partitions = 0;
    float covering_search_time = 0;
    float covering_eval_time = 0;

    void add_covering_search(float search_time, size_t partitions) {
        covering_searches++;
        covering_partitions += partitions;
        covering_search_time += search_time;
    }

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
        if (covering_searches > 0) {
            std::cout << "Covering searches: " << covering_searches << std::endl << std::flush;
            std::cout << "Covering search time: " << covering_search_time << std::endl << std::flush;
            std::cout << "Covering search average num. subindexes: " << (float) covering_partitions / covering_searches << std::endl << std::flush;
            std::cout << "------------------------------------------------------"  << std::endl << std::flush;
        }
        std::cout << "Covering eval time: " << covering_eval_time << std::endl << std::flush;
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
        covering_searches = 0;
        covering_partitions = 0;
        covering_search_time = 0;
        covering_eval_time = 0;
    }
};

template<typename T>
std::unordered_set<T> unordered_set_intersection(
    const std::unordered_set<T>& a,
    const std::unordered_set<T>& b) {

    // Always iterate through the smaller set for optimization
    if (a.size() > b.size()) {
        return unordered_set_intersection(b, a);
    }

    std::unordered_set<T> result;
    for (const auto& elem : a) {
        if (b.count(elem)) {
            result.insert(elem);
        }
    }
    return result;
}



inline size_t downscaled_M(size_t cardinality, size_t root_cardinality, size_t M) {
    // M is rounded to the nearest multiple of 4
    return std::max<size_t>(4, static_cast<size_t>(std::round((log10(cardinality) / log10(root_cardinality) * M) / 4) * 4));
}

inline size_t downscaled_ef_search(size_t cardinality, size_t root_cardinality, size_t ef_search, size_t k) {
    return std::max(k, static_cast<size_t>(log10(cardinality) / log10(root_cardinality) * ef_search));
}

template<typename dist_t, typename data_t>
class PartitionedHNSWNode {
public:
    SpaceInterface<dist_t>* _space;
    HierarchicalNSW<dist_t>* _hnsw;
    Predicate _predicate;
    std::vector<PartitionedHNSWNode<dist_t, data_t>*> _children;
    int _id;

    PartitionedHNSWNode(Predicate predicate, data_t* data, SpaceInterface<dist_t>* space, PartitionedIndexParams params)
        : _space(space), _predicate(predicate) {
        int M = params.enable_heterogeneous_indexing
            ? downscaled_M(predicate.cardinality(), params.dataset_size, params.M)
            : params.M;
        _hnsw = new HierarchicalNSW<dist_t>(_space, predicate.cardinality(), M, params.ef_construction);
        auto tmp_vec = predicate.matching_points();
        ParallelFor(0, tmp_vec.size(), params.num_threads, [&](size_t row, size_t) {
            _hnsw->addPoint((void*)(data + params.dim * tmp_vec[row]), tmp_vec[row]);
        });
    }

    void AddPoint(data_t* data, int id) {
        _hnsw->addPoint((void*)data, id);
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
        _historical_workload.insert(_historical_workload.end(), new_historical_workload.begin(), new_historical_workload.end());
        if (_historical_workload.size() > _index_params.historical_workload_window_size) {
            auto start_it = _historical_workload.end() - _index_params.historical_workload_window_size;
            _historical_workload.erase(_historical_workload.begin(), start_it);
        }
    }

    void updateData(
        data_t* _new_data,
        DatasetFilters* _new_dataset_filters,
        size_t num_threads
    ) {
        auto add_points = [&](PartitionedHNSWNode<dist_t, data_t>* node) {
            auto match_bv = _new_dataset_filters->query_matches(node->_predicate._query_filter);
            // node->_hnsw->resizeIndex(match_bv->cardinality() + node->_hnsw->cur_element_count);
            std::vector<uint32_t> tmp_vec;
            for (roaring::Roaring::const_iterator j = match_bv->begin(); j != match_bv->end(); j++) {
                tmp_vec.push_back(*j);
            }
            ParallelFor(0, tmp_vec.size(), num_threads, [&](size_t row, size_t) {
                node->_hnsw->addPoint((void*)(_new_data + _index_params.dim * tmp_vec[row]), _index_params.dataset_size + tmp_vec[row]);
            });
        };

        ParallelFor(0, _nodes.size() + 1, _index_params.num_threads, [&](size_t i, size_t) {
            if (i == _nodes.size()) {
                add_points(_root);
            } else {
                add_points(_nodes[i]);
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
        std::priority_queue<std::pair<dist_t, labeltype>> max_queue;
        // Iterate over all matching indices in the bitset
        for (auto idx : *query_bitset) {
            dist_t dist = _root->_hnsw->fstdistfunc_(query_data, _data + _index_params.dim * idx, _root->_hnsw->dist_func_param_);
            if (max_queue.size() < k) {
                max_queue.emplace(dist, idx);
            } else if (dist < max_queue.top().first) {
                max_queue.pop();
                max_queue.emplace(dist, idx);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        _index_counters.add_bruteforce_search(std::chrono::duration<double>(end - start).count(), query_cardinality);
        return max_queue;
    }

    std::priority_queue<std::pair<dist_t, labeltype>>
    searchKnnCovering(const void *query_data, size_t k, roaring::Roaring* query_bitset, std::vector<PartitionedHNSWNode<dist_t, data_t>*> indexes) {
        auto start = std::chrono::high_resolution_clock::now();
        BitMapFilter QueryBitset(query_bitset);
        std::unordered_map<hnswlib::labeltype, dist_t> labels;

        // Search in each selected index
        for (auto index : indexes) {
            std::priority_queue<std::pair<dist_t, hnswlib::labeltype>> result = index->_hnsw->searchKnn(query_data, k, &QueryBitset);
            for (; !result.empty(); result.pop()) {
                hnswlib::labeltype label = result.top().second;
                dist_t dist = result.top().first;
                labels[label] = dist;
            }
        }
    
        // Convert unordered_map to vector of pairs
        std::vector<std::pair<hnswlib::labeltype, dist_t>> vec(labels.begin(), labels.end());
        
        // Sort vector by the value (second element of the pair)
        std::sort(vec.begin(), vec.end(),
            [](const std::pair<hnswlib::labeltype, dist_t>& a, const std::pair<hnswlib::labeltype, dist_t>& b) {
                return a.second < b.second;
            }
        );

        // Retrieve top-k reranked results
        std::priority_queue<std::pair<dist_t, hnswlib::labeltype>> res;
        for (size_t j = 0; j < k; j++) {
            res.push(std::make_pair(vec[j].second, vec[j].first));
        }
        auto end = std::chrono::high_resolution_clock::now();
        _index_counters.add_covering_search(std::chrono::duration<double>(end - start).count(), indexes.size());
        return res;
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

        // Find best partition (oracle or closest ancestor)
        PartitionedHNSWNode<dist_t, data_t>* best_partition = _root;
        size_t best_size = _root->_predicate.cardinality();

        start = std::chrono::high_resolution_clock::now();
        auto it = _node_map.find(query_predicate._query_filter);
        if (it != _node_map.end()) {
            best_partition = it->second;
            best_size = best_partition->_predicate.cardinality();
        } else {
            std::queue<PartitionedHNSWNode<dist_t, data_t>*> bfs_queue;
            std::unordered_set<int> visited;
            bfs_queue.push(_root);
            while (!bfs_queue.empty()) {
                auto* cur = bfs_queue.front();
                bfs_queue.pop();
                visited.insert(cur->_id);

                if (cur->_predicate.cardinality() < best_size) {
                    best_size = cur->_predicate.cardinality();
                    best_partition = cur;
                }

                for (auto* child : cur->_children) {
                    if (visited.count(child->_id) == 0 &&
                        child->_predicate.cardinality() >= query_predicate.cardinality() &&
                        query_predicate.is_logical_subset(child->_predicate)) {
                        bfs_queue.push(child);
                    }
                }
            }
        }
        end = std::chrono::high_resolution_clock::now();
        _index_counters.predicate_comp_time += std::chrono::duration<double>(end - start).count();

        // Compute costs of different search strategies
        float cur_upward_search_cost = upward_search_cost(best_size, query_predicate.cardinality());
        float cur_bf_search_cost = bf_search_cost(query_predicate.cardinality());

        if (_index_params.enable_multipartition_search) {
            // explore options for multi-partition search
            roaring::Roaring to_cover(*query_predicate._bitvector);
            std::vector<PartitionedHNSWNode<dist_t, data_t>*> indexes;
            std::unordered_set<int> already_selected;
            float covering_search_cost = 0;

            start = std::chrono::high_resolution_clock::now();
            while (to_cover.cardinality() > 0) {
                PartitionedHNSWNode<dist_t, data_t>* best_partition = nullptr;
                float best_ratio = std::numeric_limits<float>::max();
                float best_search_cost = -1;
                int best_idx = -1;
                for (auto* cur_node : _nodes) {
                    if (already_selected.count(cur_node->_id)) continue;
                    if (query_predicate.is_logical_subset(cur_node->_predicate)) {
                        already_selected.insert(cur_node->_id);
                        continue;
                    }
                    if (to_cover.intersect(*cur_node->_predicate._bitvector)) {
                        int intersect_sel = query_predicate._bitvector->and_cardinality(*cur_node->_predicate._bitvector);
                        int intersect_size = to_cover.and_cardinality(*cur_node->_predicate._bitvector);
                        float search_cost = upward_search_cost(cur_node->_predicate.cardinality(), intersect_sel);
                        float ratio = search_cost / intersect_size;
                        if (ratio < best_ratio) {
                            best_ratio = ratio;
                            best_search_cost = search_cost;
                            best_idx = cur_node->_id;
                            best_partition = cur_node;
                        }
                    }
                }
                if (best_idx == -1) break;
                to_cover -= *best_partition->_predicate._bitvector;
                covering_search_cost += best_search_cost;
                already_selected.insert(best_idx);
                indexes.push_back(best_partition);
            }
            // skip if partitions don't cover candidate
            if (to_cover.cardinality() > 0) {
                covering_search_cost = std::numeric_limits<float>::max();
            }
            end = std::chrono::high_resolution_clock::now();
            _index_counters.covering_eval_time += std::chrono::duration<double>(end - start).count();

            if (cur_bf_search_cost <= covering_search_cost && cur_bf_search_cost <= cur_upward_search_cost) {
                return searchKnnBf(query_data, k, query_predicate._bitvector, query_predicate.cardinality());
            }
            if (covering_search_cost <= cur_bf_search_cost && covering_search_cost <= cur_upward_search_cost) {
                return searchKnnCovering(query_data, k, query_predicate._bitvector, indexes);
            }
        } else {
            if (cur_bf_search_cost <= cur_upward_search_cost) {
                return searchKnnBf(query_data, k, query_predicate._bitvector, query_predicate.cardinality());
            }
        }

        start = std::chrono::high_resolution_clock::now();
        auto res = best_partition->_hnsw->searchKnn(query_data, k, &QueryBitset);
        end = std::chrono::high_resolution_clock::now();

        // Update counters
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
        // Tally query filters and their counts
        auto filter_counts = tally_query_filters(_historical_workload, _dataset_filters, _index_params.bitvector_cutoff);

        // Convert each filter/count pair into a Predicate/count pair
        std::vector<std::pair<Predicate, size_t>> result;
        result.reserve(filter_counts.size());
        for (const auto& fc : filter_counts) {
            result.emplace_back(Predicate(_dataset_filters, fc.first), fc.second);
        }

        // Sort by predicate cardinality
        std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) {
            return a.first.cardinality() < b.first.cardinality();
        });

        std::cout << "candidate partitions: " << result.size() << std::endl << std::flush;
        return result;
    }

    // Find (IDs of) partitions to construct based on historical workload.
    std::vector<int> find_partitions(const std::vector<std::pair<Predicate, size_t>>& hw_preds) {
        // Find edges
        std::unordered_map<int, std::unordered_set<int>> parent_set;
        std::unordered_map<int, std::unordered_set<int>> child_set;
        int num_edges = 0;

        // Collect all edges in a single vector for simplicity
        std::vector<std::pair<size_t, size_t>> edges;
        ParallelFor(0, hw_preds.size(), _index_params.num_threads, [&](size_t i, size_t) {
            for (size_t j = i; j < hw_preds.size(); j++) {
                if (hw_preds[i].first.is_logical_subset(hw_preds[j].first)) {
                    #pragma omp critical
                    edges.emplace_back(i, j);
                }
            }
        });

        // Populate parent_set and child_set from edges
        for (const auto& e : edges) {
            num_edges++;
            int child = static_cast<int>(e.first);
            int parent = static_cast<int>(e.second);
            parent_set[child].insert(parent);
            child_set[parent].insert(child);
        }

        // Submodular optimization
        size_t total_vecs = 0;
        size_t total_budget = static_cast<size_t>(
            (static_cast<float>(_index_params.index_vector_budget) / _index_params.dataset_size) *
            scaled_partition_size(_index_params.dataset_size)
        );

        std::vector<float> best_costs(hw_preds.size());
        std::vector<bool> dirty(hw_preds.size(), false);

        for (size_t i = 0; i < hw_preds.size(); i++) {
            best_costs[i] = std::min(
                bf_search_cost(hw_preds[i].first.cardinality()),
                root_search_cost(hw_preds[i].first.cardinality())
            );
        }

        // Compute initial round of marginal benefits
        using QueueElem = std::pair<float, int>;
        auto cmp = [](const QueueElem& a, const QueueElem& b) { return a.first < b.first; };
        std::priority_queue<QueueElem, std::vector<QueueElem>, decltype(cmp) > max_queue(cmp);

        for (size_t i = 0; i < hw_preds.size(); i++) {
            float ratio_sum = 0.0f;
            for (const auto& child : child_set[i]) {
                float benefit = best_costs[child] - upward_search_cost(hw_preds[child].first.cardinality(), hw_preds[i].first.cardinality());
                ratio_sum += std::max(0.0f, benefit) * hw_preds[child].second;
            }
            float scaled = scaled_partition_size(hw_preds[i].first.cardinality());
            max_queue.emplace(scaled > 0 ? ratio_sum / scaled : 0.0f, static_cast<int>(i));
        }

        std::cout << "edges:" << num_edges << std::endl << std::flush;
        std::cout << "Initialized submodular optimizer." << std::endl << std::flush;

        std::unordered_set<int> selected;
        std::cout << "total vecs:" << total_vecs;
        std::cout << "total budget:" << total_budget;

        while (total_vecs < total_budget && selected.size() < hw_preds.size() && !max_queue.empty()) {
            int node = max_queue.top().second;
            max_queue.pop();

            // Node is already selected
            if (selected.count(node)) continue;

            // popped benefit of current node is inaccurate (due to a parent or child being selected); recompute it and add it back to the pqueue.
            if (dirty[node]) {
                float ratio_sum = 0.0f;
                for (const auto& child : child_set[node]) {
                    float benefit = best_costs[child] - upward_search_cost(hw_preds[child].first.cardinality(), hw_preds[node].first.cardinality());
                    ratio_sum += std::max(0.0f, benefit) * hw_preds[child].second;
                }
                float scaled = scaled_partition_size(hw_preds[node].first.cardinality());
                max_queue.emplace(scaled > 0 ? ratio_sum / scaled : 0.0f, node);
                dirty[node] = false;
                continue;
            }

            // Select the node and mark the costs of all its children and parents as dirty.
            for (const auto& child : child_set[node]) {
                if (!selected.count(child)) dirty[child] = true;
            }
            for (const auto& parent : parent_set[node]) {
                if (!selected.count(parent)) dirty[parent] = true;
            }

            // Update the scores of all queries potentially handled at this node.
            for (const auto& child : child_set[node]) {
                best_costs[child] = std::min(
                    upward_search_cost(hw_preds[child].first.cardinality(), hw_preds[node].first.cardinality()),
                    best_costs[child]
                );
            }

            selected.insert(node);
            total_vecs += scaled_partition_size(hw_preds[node].first.cardinality());
        }

        std::cout << "Number of partitions: " << selected.size() << std::endl << std::flush;

        // Find constructed and deleted subindexes.
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
        std::vector<int> selected_partitions(selected.begin(), selected.end());
        int num_final_edges = 0;
        for (const auto& i : selected_partitions) {
            for (const auto& child : child_set[i]) {
                if (selected.count(child)) {
                    num_final_edges++;
                }
            }
        }
        std::cout << "Number of edges: " << num_edges << std::endl << std::flush;

        return selected_partitions;
    }

    void construct_partitions(std::vector<std::pair<Predicate, size_t>> hw_preds, std::vector<int> selected_partitions) {
        std::unordered_set<QueryFilter, QueryFilterHash> partitions_to_construct;
        for (int idx : selected_partitions)
            partitions_to_construct.insert(hw_preds[idx].first._query_filter);

        // Delete outdated partitions
        std::vector<QueryFilter> to_delete;
        for (const auto& it : _node_map)
            if (!partitions_to_construct.count(it.first))
                to_delete.push_back(it.first);

        for (const auto& key : to_delete) {
            delete _node_map[key];
            _node_map.erase(key);
        }
        std::cout << "Total partitions:" << selected_partitions.size() << std::endl << std::flush;
        std::cout << "Deleted partitions: " << to_delete.size() << std::endl << std::flush;

        // Find new partitions to construct
        std::unordered_set<int> new_selected_partitions;
        size_t new_vec_count = 0;
        for (int idx : selected_partitions) {
            if (!_node_map.count(hw_preds[idx].first._query_filter)) {
                new_selected_partitions.insert(idx);
                new_vec_count += hw_preds[idx].first.cardinality();
            }
        }

        std::vector<int> new_partition_indices(new_selected_partitions.begin(), new_selected_partitions.end());
        std::cout << "New partitions to construct: " << new_partition_indices.size() << std::endl << std::flush;
        std::cout << "New partitions vectors: " << new_vec_count << std::endl << std::flush;

        // Construct root and new partitions
        if (!has_root) {
            has_root = true;
            ParallelFor(0, new_partition_indices.size() + 1, _index_params.num_threads, [&](size_t i, size_t) {
                if (i == new_partition_indices.size() && _root == nullptr) {
                    _root = new PartitionedHNSWNode<dist_t, data_t>(Predicate(_index_params.dataset_size), _data, _space, _index_params);
                } else if (i < new_partition_indices.size()) {
                    int idx = new_partition_indices[i];
                    _node_map[hw_preds[idx].first._query_filter] = new PartitionedHNSWNode<dist_t, data_t>(hw_preds[idx].first, _data, _space, _index_params);
                }
            });
        } else {
            ParallelFor(0, new_partition_indices.size(), _index_params.num_threads, [&](size_t i, size_t) {
                int idx = new_partition_indices[i];
                _node_map[hw_preds[idx].first._query_filter] = new PartitionedHNSWNode<dist_t, data_t>(hw_preds[idx].first, _data, _space, _index_params);
            });
        }

        std::cout << "Updated partitions." << std::endl << std::flush;

        // Create nodes vector
        _nodes.clear();
        for (const auto& it : _node_map) {
            _nodes.push_back(it.second);
        }

        std::sort(_nodes.begin(), _nodes.end(), [](auto* a, auto* b) {
            return a->_predicate.cardinality() < b->_predicate.cardinality();
        });
        std::cout << "Done." << std::endl << std::flush;

        // Assign IDs
        for (size_t i = 0; i < _nodes.size(); i++) {
            _nodes[i]->_id = i;
        }
        if (_root) {
            _root->_id = -1;
        }
    }

    void create_hasse_diagram() {
        // Find edges in constructed partitions
        std::unordered_map<int, std::unordered_set<int>> parent_set, child_set;
        int num_edges = 0;

        std::vector<std::pair<size_t, size_t>> edges;
        for (size_t i = 0; i < _nodes.size(); i++) {
            for (size_t j = i + 1; j < _nodes.size(); j++) {
                if (_nodes[i]->_predicate.is_logical_subset(_nodes[j]->_predicate)) {
                    edges.emplace_back(i, j);
                }
            }
        }

        for (auto& e : edges) {
            num_edges++;
            parent_set[e.first].insert(e.second);
            child_set[e.second].insert(e.first);
        }

        // Clear children vectors in nodes
        for (auto* node : _nodes) node->_children.clear();
        _root->_children.clear();

        // Fill children vectors in nodes.
        // Note: _nodes is already sorted w.r.t. ascending cardinality.
        size_t hasse_diagram_edges = 0;
        for (size_t i = 0; i < _nodes.size(); i++) {
            if (parent_set.find(i) == parent_set.end()) continue;
            for (size_t j : parent_set[i]) {
                // Check for direct relationship by ensuring no intermediate 
                bool direct = true;
                for (size_t k = i + 1; k < j; k++) {
                    if (parent_set[i].count(k) && parent_set[k].count(j)) {
                        direct = false;
                        break;
                    }
                }
                // If direct subset relationship, add as a child in the Hasse diagram
                if (direct) {
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
        float new_ef = _index_params.enable_heterogeneous_search
            ? downscaled_ef_search(parent_cardinality, _index_params.dataset_size, _ef, _k)
            : _ef;
        float new_ef_ratio = (_k + ((new_ef - _k) / _index_params.ef_search_scaling_constant)) / _k;
        return log(parent_cardinality) * pow(
            parent_cardinality / query_cardinality, _index_params.query_correlation_constant) * new_ef_ratio;
    }

    float root_search_cost(size_t query_cardinality) {
        float new_ef_ratio = (_k + ((_ef - _k) / _index_params.ef_search_scaling_constant)) / _k;
        return log(_index_params.dataset_size) * pow(
            _index_params.dataset_size / query_cardinality, _index_params.query_correlation_constant) * new_ef_ratio;
    }

    size_t scaled_partition_size(size_t cardinality) {
        // Compute M: empirical equation from pinecone (https://www.pinecone.io/learn/series/faiss/hnsw/)
        size_t M = _index_params.enable_heterogeneous_indexing
            ? downscaled_M(cardinality, _index_params.dataset_size, _index_params.M)
            : _index_params.M;
        return cardinality * (M + 50) / 82;
    }
};

} // namespace hnswlib