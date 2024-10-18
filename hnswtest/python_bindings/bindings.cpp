#include <iostream>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "partitioned_hnsw.h"
#include "hnswalg.h"
#include "filters.h"
#include "space_l2.h"
#include <limits>
#include <thread>
#include <atomic>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <utility>
#include <stdint.h>
#include <assert.h>
#include <chrono>
#include <immintrin.h>
#include <omp.h>

#include "roaring.hh"
#include "roaring.c"
#include "xxhash.h"
#include <faiss/IndexACORN.h>
// #include "xxhash.c"

// Caps includes
#include "caps_include/FilterIndex.h"
#include "caps_include/FilterIndex.cpp"

// NHQ includes
#include "nhq_include/index_random.h"
#include "nhq_include/index_random.cpp"
#include "nhq_include/index_graph.h"
#include "nhq_include/index_graph.cpp"
#include "nhq_include/util.h"

namespace py = pybind11;
using namespace pybind11::literals;  // needed to bring in _a literal

// reads 0 <= d < 4 floats as __m128
static inline __m128 masked_read(int d, const float* x) {
    assert(0 <= d && d < 4);
    ALIGNED(16) float buf[4] = {0, 0, 0, 0};
    switch (d) {
        case 3:
            buf[2] = x[2];
        case 2:
            buf[1] = x[1];
        case 1:
            buf[0] = x[0];
    }
    return _mm_load_ps(buf);
    // cannot use AVX2 _mm_mask_set1_epi32
}

float Faiss_fvec_L2sqr(const float* x, const float* y, size_t d) {
    __m256 msum1 = _mm256_setzero_ps();

    while (d >= 8) {
        __m256 mx = _mm256_loadu_ps(x);
        x += 8;
        __m256 my = _mm256_loadu_ps(y);
        y += 8;
        const __m256 a_m_b1 = _mm256_sub_ps(mx, my);
        msum1 = _mm256_add_ps(msum1, _mm256_mul_ps(a_m_b1, a_m_b1));
        d -= 8;
    }

    __m128 msum2 = _mm256_extractf128_ps(msum1, 1);
    msum2 = _mm_add_ps(msum2, _mm256_extractf128_ps(msum1, 0));

    if (d >= 4) {
        __m128 mx = _mm_loadu_ps(x);
        x += 4;
        __m128 my = _mm_loadu_ps(y);
        y += 4;
        const __m128 a_m_b1 = _mm_sub_ps(mx, my);
        msum2 = _mm_add_ps(msum2, _mm_mul_ps(a_m_b1, a_m_b1));
        d -= 4;
    }

    if (d > 0) {
        __m128 mx = masked_read(d, x);
        __m128 my = masked_read(d, y);
        __m128 a_m_b1 = _mm_sub_ps(mx, my);
        msum2 = _mm_add_ps(msum2, _mm_mul_ps(a_m_b1, a_m_b1));
    }

    msum2 = _mm_hadd_ps(msum2, msum2);
    msum2 = _mm_hadd_ps(msum2, msum2);
    return _mm_cvtss_f32(msum2);
}

class HierarchicalIndexUint8 {
 public:
    static const int ser_version = 1;  // serialization version

    uint8_t* _data;
    hnswlib::PartitionedHNSW<int, uint8_t>* alg;
    hnswlib::DatasetFilters* dataset_filters;

    HierarchicalIndexUint8(
        std::string filename,
        std::string filter_filename,
        const std::vector<hnswlib::QueryFilter>& historical_workload,
        size_t dataset_size,
        size_t dim,
        size_t M,
        size_t ef_construction,
        size_t index_vector_budget,
        size_t bitvector_cutoff,
        size_t historical_workload_window_size,
        bool enable_heterogeneous_indexing,
        size_t num_threads
    ) {
        auto start = std::chrono::high_resolution_clock::now();
        // setup index params
        hnswlib::PartitionedIndexParams index_params{
            dataset_size, dim, M, ef_construction, index_vector_budget, bitvector_cutoff, historical_workload_window_size, enable_heterogeneous_indexing, num_threads};
        // setup filters
        dataset_filters = new hnswlib::DatasetFilters(fopen(filter_filename.c_str(), "rb"), num_threads);
        dataset_filters->transpose_inplace();
        dataset_filters->make_bvs();

        // setup data
        std::ifstream reader(filename);
        assert(reader.is_open());
        size_t num_points;
        size_t d;
        reader.read((char*)(&num_points), sizeof(unsigned int));
        reader.read((char*)(&d), sizeof(unsigned int));

        _data = new uint8_t[dim * dataset_size];
        reader.read((char*)_data, sizeof(uint8_t) * dim * dataset_size);

        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Time to read data: " << std::chrono::duration<double>(end - start).count() << std::endl << std::flush;

        hnswlib::L2SpaceX* space = new hnswlib::L2SpaceX(dim / 4);
        alg = new hnswlib::PartitionedHNSW<int, uint8_t>(
            _data,
            space,
            dataset_filters,
            index_params,
            historical_workload
        );
    }

    void update_index(const std::vector<hnswlib::QueryFilter>& historical_workload) {
        alg->updateIndexWorkload(historical_workload);
        alg->fitIndex();
    }

    py::array_t<unsigned int> batch_filter_search(
     py::array_t<uint8_t, py::array::c_style | py::array::forcecast>& queries,
     const std::vector<hnswlib::QueryFilter>& filters, uint64_t num_queries,
     uint64_t knn, size_t ef_search, uint64_t num_threads) {
    py::array_t<unsigned int> ids({num_queries, knn});
    
    auto start = std::chrono::high_resolution_clock::now();
    // Set ef_search of partitions
    alg->setEf(ef_search);
    std::vector<hnswlib::Predicate> predicate_arr;
    for (int i = 0; i < filters.size(); i++) {
        predicate_arr.push_back(hnswlib::Predicate(dataset_filters, filters[i]));
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Time construct predicates: " << std::chrono::duration<double>(end - start).count() << std::endl << std::flush;

    start = std::chrono::high_resolution_clock::now();
    hnswlib::ParallelFor(0, filters.size(), num_threads, [&](size_t i, size_t threadId) {
        auto results = alg->searchKnn(queries.data(i), knn, predicate_arr[i]);
        for (size_t j = 0; j < knn; j++) {
            if (!results.empty()) {
                ids.mutable_data(i)[j] = results.top().second;
                // std::cout << results.top().second << ' ';
                results.pop();
            } else {
                ids.mutable_data(i)[j] = 0;
            }
        }
    }); 
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Time serve queries: " << std::chrono::duration<double>(end - start).count() << std::endl << std::flush;
    alg->printTally();

    return ids;
  }
};


class HierarchicalIndexFloat {
 public:
    static const int ser_version = 1;  // serialization version

    float* _data;
    hnswlib::PartitionedHNSW<float, float>* alg;
    hnswlib::DatasetFilters* dataset_filters;
    size_t _dataset_size;

    // update test
    float* _new_data;
    hnswlib::DatasetFilters* new_dataset_filters;

    HierarchicalIndexFloat(
        std::string filename,
        std::string filter_filename,
        const std::vector<hnswlib::QueryFilter>& historical_workload,
        size_t dataset_size,
        size_t dim,
        size_t M,
        size_t ef_construction,
        size_t index_vector_budget,
        size_t bitvector_cutoff,
        size_t historical_workload_window_size,
        bool enable_heterogeneous_indexing,
        size_t num_threads
    ) : _dataset_size(dataset_size) {
        auto start = std::chrono::high_resolution_clock::now();
        // setup index params
        hnswlib::PartitionedIndexParams index_params{
            dataset_size, dim, M, ef_construction, index_vector_budget, bitvector_cutoff, historical_workload_window_size, enable_heterogeneous_indexing, num_threads};
        // setup filters
        dataset_filters = new hnswlib::DatasetFilters(fopen(filter_filename.c_str(), "rb"), num_threads);
        dataset_filters->transpose_inplace();
        dataset_filters->make_bvs();

        // setup data
        std::ifstream reader(filename);
        assert(reader.is_open());
        size_t num_points;
        size_t d;
        reader.read((char*)(&num_points), sizeof(unsigned int));
        reader.read((char*)(&d), sizeof(unsigned int));

        _data = new float[dim * dataset_size];
        reader.read((char*)_data, sizeof(float) * dim * dataset_size);

        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Time to read data: " << std::chrono::duration<double>(end - start).count() << std::endl << std::flush;

        hnswlib::L2Space* space = new hnswlib::L2Space(dim);
        alg = new hnswlib::PartitionedHNSW<float, float>(
            _data,
            space,
            dataset_filters,
            index_params,
            historical_workload
        );
    }

    void update_index(const std::vector<hnswlib::QueryFilter>& historical_workload) {
        alg->updateIndexWorkload(historical_workload);
        alg->fitIndex();
    }

    void update_data(        
        std::string new_filename,
        std::string new_filter_filename,
        size_t update_size,
        size_t update_dim,
        size_t num_threads) {
        auto start = std::chrono::high_resolution_clock::now();
        // setup filters
        new_dataset_filters = new hnswlib::DatasetFilters(fopen(new_filter_filename.c_str(), "rb"), num_threads);
        new_dataset_filters->transpose_inplace();
        new_dataset_filters->make_bvs();

        // setup data
        std::ifstream reader(new_filename);
        assert(reader.is_open());
        size_t num_points;
        size_t d;
        reader.read((char*)(&num_points), sizeof(unsigned int));
        reader.read((char*)(&d), sizeof(unsigned int));

        _new_data = new float[update_dim * update_size];
        reader.read((char*)_data, sizeof(float) * update_dim * update_size);
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Time to read update data: " << std::chrono::duration<double>(end - start).count() << std::endl << std::flush;

        start = std::chrono::high_resolution_clock::now();
        alg->updateData(_new_data, new_dataset_filters, num_threads);
        end = std::chrono::high_resolution_clock::now();
        std::cout << "Time to update index: " << std::chrono::duration<double>(end - start).count() << std::endl << std::flush;
    }

    py::array_t<unsigned int> batch_filter_search(
     py::array_t<float, py::array::c_style | py::array::forcecast>& queries,
     const std::vector<hnswlib::QueryFilter>& filters, uint64_t num_queries,
     uint64_t knn, size_t ef_search, uint64_t num_threads) {
    py::array_t<unsigned int> ids({num_queries, knn});
    float maxsel_search_time = 0;
    auto start = std::chrono::high_resolution_clock::now();
    // Set ef_search of partitions
    alg->setEf(ef_search);
    std::vector<hnswlib::Predicate> predicate_arr;
    for (int i = 0; i < filters.size(); i++) {
        predicate_arr.push_back(hnswlib::Predicate(dataset_filters, filters[i]));
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Time construct predicates: " << std::chrono::duration<double>(end - start).count() << std::endl << std::flush;

    start = std::chrono::high_resolution_clock::now();
    hnswlib::ParallelFor(0, filters.size(), num_threads, [&](size_t i, size_t threadId) {
        // auto start2 = std::chrono::high_resolution_clock::now();
        auto results = alg->searchKnn(queries.data(i), knn, predicate_arr[i]);
        // auto end2 = std::chrono::high_resolution_clock::now();
        for (size_t j = 0; j < knn; j++) {
            if (!results.empty()) {
                ids.mutable_data(i)[j] = results.top().second;
                // std::cout << results.top().second << ' ';
                results.pop();
            } else {
                ids.mutable_data(i)[j] = 0;
            }
        }
        // if (predicate_arr[i].cardinality() == _dataset_size) {
        // maxsel_search_time += std::chrono::duration<double>(end2 - start2).count();
        // }
    }); 
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Time serve queries: " << std::chrono::duration<double>(end - start).count() << std::endl << std::flush;
    std::cout << "max selectivity query time: " << maxsel_search_time << std::endl << std::flush;
    alg->printTally();

    return ids;
  }
};


class PreFilterUint8 {
 public:
    static const int ser_version = 1;  // serialization version

    uint8_t* _data;
    hnswlib::DatasetFilters* dataset_filters;
    hnswlib::HierarchicalNSW<int>* _hnsw;
    size_t _dataset_size;
    size_t _dim;

    PreFilterUint8(
        std::string filename,
        std::string filter_filename,
        size_t dataset_size,
        size_t dim,
        size_t num_threads
    ) : _dataset_size(dataset_size), _dim(dim) {
        auto start = std::chrono::high_resolution_clock::now();
        // setup filters
        dataset_filters = new hnswlib::DatasetFilters(fopen(filter_filename.c_str(), "rb"), num_threads);
        dataset_filters->transpose_inplace();
        dataset_filters->make_bvs();

        // setup data
        std::ifstream reader(filename);
        assert(reader.is_open());
        size_t num_points;
        size_t d;
        reader.read((char*)(&num_points), sizeof(unsigned int));
        reader.read((char*)(&d), sizeof(unsigned int));

        _data = new uint8_t[dim * dataset_size];
        reader.read((char*)_data, sizeof(uint8_t) * dim * dataset_size);

        auto end = std::chrono::high_resolution_clock::now();

        // Just here for the distance calcs
        hnswlib::L2SpaceX* space = new hnswlib::L2SpaceX(dim / 4);
        _hnsw = new hnswlib::HierarchicalNSW<int>(space, 10, 16, 40);

        std::cout << "Time to read data: " << std::chrono::duration<double>(end - start).count() << std::endl << std::flush;
    }

    py::array_t<unsigned int> batch_filter_search(
     py::array_t<uint8_t, py::array::c_style | py::array::forcecast>& queries,
     const std::vector<hnswlib::QueryFilter>& filters, uint64_t num_queries,
     uint64_t knn, uint64_t num_threads) {
    py::array_t<unsigned int> ids({num_queries, knn});

    auto start = std::chrono::high_resolution_clock::now();
    // Set ef_search of partitions
    std::vector<hnswlib::Predicate> predicate_arr;
    for (int i = 0; i < filters.size(); i++) {
        predicate_arr.push_back(hnswlib::Predicate(dataset_filters, filters[i]));
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Time construct predicates: " << std::chrono::duration<double>(end - start).count() << std::endl << std::flush;

    size_t bruteforce_distance_comps = 0;
    start = std::chrono::high_resolution_clock::now();
    hnswlib::ParallelFor(0, filters.size(), num_threads, [&](size_t i, size_t threadId) {
        roaring::Roaring& deref = *predicate_arr[i]._bitvector;
        std::priority_queue<std::pair<int, size_t>> max_priority_queue;
        for (roaring::Roaring::const_iterator j = deref.begin(); j != deref.end(); j++) {
            int dist = _hnsw->fstdistfunc_(queries.data(i), (_data + _dim * *j), _hnsw->dist_func_param_);
            if (max_priority_queue.size() < knn) {
                max_priority_queue.push(std::make_pair(dist, *j));
            } else if (dist < max_priority_queue.top().first) {
                max_priority_queue.pop();
                max_priority_queue.push(std::make_pair(dist, *j));
            }
        }
        bruteforce_distance_comps += predicate_arr[i].cardinality();
        for (size_t j = 0; j < knn; j++) {
            if (!max_priority_queue.empty()) {
                ids.mutable_data(i)[j] = max_priority_queue.top().second;
                // std::cout << results.top().second << ' ';
                max_priority_queue.pop();
            } else {
                ids.mutable_data(i)[j] = 0;
            }
        }
    }); 
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Time serve queries: " << std::chrono::duration<double>(end - start).count() << std::endl << std::flush;
    std::cout << "Num distance comps: " << bruteforce_distance_comps << std::endl << std::flush;

    return ids;
  }
};

class PreFilterFloat {
 public:
    static const int ser_version = 1;  // serialization version

    float* _data;
    hnswlib::DatasetFilters* dataset_filters;
    hnswlib::HierarchicalNSW<float>* _hnsw;
    size_t _dataset_size;
    size_t _dim;

    PreFilterFloat(
        std::string filename,
        std::string filter_filename,
        size_t dataset_size,
        size_t dim,
        size_t num_threads
    ) : _dataset_size(dataset_size), _dim(dim) {
        auto start = std::chrono::high_resolution_clock::now();
        // setup filters
        dataset_filters = new hnswlib::DatasetFilters(fopen(filter_filename.c_str(), "rb"), num_threads);
        dataset_filters->transpose_inplace();
        dataset_filters->make_bvs();

        // setup data
        std::ifstream reader(filename);
        assert(reader.is_open());
        size_t num_points;
        size_t d;
        reader.read((char*)(&num_points), sizeof(unsigned int));
        reader.read((char*)(&d), sizeof(unsigned int));

        _data = new float[dim * dataset_size];
        reader.read((char*)_data, sizeof(float) * dim * dataset_size);

        auto end = std::chrono::high_resolution_clock::now();

        // Just here for the distance calcs
        hnswlib::L2Space* space = new hnswlib::L2Space(dim);
        _hnsw = new hnswlib::HierarchicalNSW<float>(space, 10, 16, 40);

        std::cout << "Time to read data: " << std::chrono::duration<double>(end - start).count() << std::endl << std::flush;
    }

    py::array_t<unsigned int> batch_filter_search(
     py::array_t<float, py::array::c_style | py::array::forcecast>& queries,
     const std::vector<hnswlib::QueryFilter>& filters, uint64_t num_queries,
     uint64_t knn, uint64_t num_threads) {
    py::array_t<unsigned int> ids({num_queries, knn});

    auto start = std::chrono::high_resolution_clock::now();
    // Set ef_search of partitions
    std::vector<hnswlib::Predicate> predicate_arr;
    for (int i = 0; i < filters.size(); i++) {
        predicate_arr.push_back(hnswlib::Predicate(dataset_filters, filters[i]));
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Time construct predicates: " << std::chrono::duration<double>(end - start).count() << std::endl << std::flush;

    size_t bruteforce_distance_comps = 0;
    start = std::chrono::high_resolution_clock::now();
    hnswlib::ParallelFor(0, filters.size(), num_threads, [&](size_t i, size_t threadId) {
        roaring::Roaring& deref = *predicate_arr[i]._bitvector;
        std::priority_queue<std::pair<float, size_t>> max_priority_queue;
        for (roaring::Roaring::const_iterator j = deref.begin(); j != deref.end(); j++) {
            float dist = Faiss_fvec_L2sqr(queries.data(i), (_data + _dim * *j), _dim);
            // float dist = _hnsw->fstdistfunc_(queries.data(i), (_data + _dim * *j), _hnsw->dist_func_param_);
            if (max_priority_queue.size() < knn) {
                max_priority_queue.push(std::make_pair(dist, *j));
            } else if (dist < max_priority_queue.top().first) {
                max_priority_queue.pop();
                max_priority_queue.push(std::make_pair(dist, *j));
            }
        }
        bruteforce_distance_comps += predicate_arr[i].cardinality();
        for (size_t j = 0; j < knn; j++) {
            if (!max_priority_queue.empty()) {
                ids.mutable_data(i)[j] = max_priority_queue.top().second;
                // std::cout << results.top().second << ' ';
                max_priority_queue.pop();
            } else {
                ids.mutable_data(i)[j] = 0;
            }
        }
    }); 
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Time serve queries: " << std::chrono::duration<double>(end - start).count() << std::endl << std::flush;
    std::cout << "Num distance comps: " << bruteforce_distance_comps << std::endl << std::flush;

    return ids;
  }
};


class HNSWBaseUint8 {
 public:
    static const int ser_version = 1;  // serialization version

    uint8_t* _data;
    hnswlib::SpaceInterface<int>* _space;
    hnswlib::HierarchicalNSW<int>* _hnsw;
    hnswlib::DatasetFilters* dataset_filters;

    HNSWBaseUint8(
        std::string filename,
        std::string filter_filename,
        size_t dataset_size,
        size_t dim,
        size_t M,
        size_t ef_construction,
        size_t num_threads
    ) {
        auto start = std::chrono::high_resolution_clock::now();
        // setup filters
        dataset_filters = new hnswlib::DatasetFilters(fopen(filter_filename.c_str(), "rb"), num_threads);
        dataset_filters->transpose_inplace();
        dataset_filters->make_bvs();

        // setup data
        std::ifstream reader(filename);
        assert(reader.is_open());
        size_t num_points;
        size_t d;
        reader.read((char*)(&num_points), sizeof(unsigned int));
        reader.read((char*)(&d), sizeof(unsigned int));

        _data = new uint8_t[dim * dataset_size];
        reader.read((char*)_data, sizeof(uint8_t) * dim * dataset_size);

        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Time to read data: " << std::chrono::duration<double>(end - start).count() << std::endl << std::flush;

        // Construct index
        start = std::chrono::high_resolution_clock::now();
        _space = new hnswlib::L2SpaceX(dim / 4);
        _hnsw = new hnswlib::HierarchicalNSW<int>(_space, dataset_size, M, ef_construction);
        hnswlib::ParallelFor(0, dataset_size, num_threads, [&](size_t row, size_t threadId) {
            _hnsw->addPoint((void*)(_data + dim * row), row);
        });
        end = std::chrono::high_resolution_clock::now();
        std::cout << "Time to build index: " << std::chrono::duration<double>(end - start).count() << std::endl << std::flush;
    }

    py::array_t<unsigned int> batch_filter_search(
     py::array_t<uint8_t, py::array::c_style | py::array::forcecast>& queries,
     const std::vector<hnswlib::QueryFilter>& filters, uint64_t num_queries,
     uint64_t knn, size_t ef_search, uint64_t num_threads) {
    py::array_t<unsigned int> ids({num_queries, knn});

    
    auto start = std::chrono::high_resolution_clock::now();
    // Set ef_search of partitions
    _hnsw->setEf(ef_search);
    std::vector<hnswlib::Predicate> predicate_arr;
    for (int i = 0; i < filters.size(); i++) {
        predicate_arr.push_back(hnswlib::Predicate(dataset_filters, filters[i]));
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Time construct predicates: " << std::chrono::duration<double>(end - start).count() << std::endl << std::flush;
    start = std::chrono::high_resolution_clock::now();
    hnswlib::ParallelFor(0, filters.size(), num_threads, [&](size_t i, size_t threadId) {
        hnswlib::BitMapFilter QueryBitset(predicate_arr[i]._bitvector);
        auto results = _hnsw->searchKnn(queries.data(i), knn, &QueryBitset);
        for (size_t j = 0; j < knn; j++) {
            if (!results.empty()) {
                ids.mutable_data(i)[j] = results.top().second;
                // std::cout << results.top().second << ' ';
                results.pop();
            } else {
                ids.mutable_data(i)[j] = 0;
            }
        }
    }); 
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Time serve queries: " << std::chrono::duration<double>(end - start).count() << std::endl << std::flush;
    std::cout << "Num distance comps: " << _hnsw->metric_distance_computations << std::endl << std::flush;
    _hnsw->metric_distance_computations = 0;

    return std::move(ids);
  }
};

class HNSWBaseFloat {
 public:
    static const int ser_version = 1;  // serialization version

    float* _data;
    hnswlib::SpaceInterface<float>* _space;
    hnswlib::HierarchicalNSW<float>* _hnsw;
    hnswlib::DatasetFilters* dataset_filters;

    HNSWBaseFloat(
        std::string filename,
        std::string filter_filename,
        size_t dataset_size,
        size_t dim,
        size_t M,
        size_t ef_construction,
        size_t num_threads
    ) {
        auto start = std::chrono::high_resolution_clock::now();
        // setup filters
        dataset_filters = new hnswlib::DatasetFilters(fopen(filter_filename.c_str(), "rb"), num_threads);
        dataset_filters->transpose_inplace();
        dataset_filters->make_bvs();

        // setup data
        std::ifstream reader(filename);
        assert(reader.is_open());
        size_t num_points;
        size_t d;
        reader.read((char*)(&num_points), sizeof(unsigned int));
        reader.read((char*)(&d), sizeof(unsigned int));

        _data = new float[dim * dataset_size];
        reader.read((char*)_data, sizeof(float) * dim * dataset_size);

        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Time to read data: " << std::chrono::duration<double>(end - start).count() << std::endl << std::flush;

        // Construct index
        start = std::chrono::high_resolution_clock::now();
        _space = new hnswlib::L2Space(dim);
        _hnsw = new hnswlib::HierarchicalNSW<float>(_space, dataset_size, M, ef_construction);
        hnswlib::ParallelFor(0, dataset_size, num_threads, [&](size_t row, size_t threadId) {
            _hnsw->addPoint((void*)(_data + dim * row), row);
        });
        end = std::chrono::high_resolution_clock::now();
        std::cout << "Time to build index: " << std::chrono::duration<double>(end - start).count() << std::endl << std::flush;
    }

    py::array_t<unsigned int> batch_filter_search(
     py::array_t<float, py::array::c_style | py::array::forcecast>& queries,
     const std::vector<hnswlib::QueryFilter>& filters, uint64_t num_queries,
     uint64_t knn, size_t ef_search, uint64_t num_threads) {
    py::array_t<unsigned int> ids({num_queries, knn});

    
    auto start = std::chrono::high_resolution_clock::now();
    // Set ef_search of partitions
    _hnsw->setEf(ef_search);
    std::vector<hnswlib::Predicate> predicate_arr;
    for (int i = 0; i < filters.size(); i++) {
        predicate_arr.push_back(hnswlib::Predicate(dataset_filters, filters[i]));
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Time construct predicates: " << std::chrono::duration<double>(end - start).count() << std::endl << std::flush;
    start = std::chrono::high_resolution_clock::now();
    hnswlib::ParallelFor(0, filters.size(), num_threads, [&](size_t i, size_t threadId) {
        // std::cout << "query " << i << std::endl;
        // std::cout << ": filters: ";
        // for (auto i: predicate_arr[i]._query_filter._filters)
        //     std::cout << i << ' ';
        // std::cout << std::endl;

        hnswlib::BitMapFilter QueryBitset(predicate_arr[i]._bitvector);
        auto results = _hnsw->searchKnn(queries.data(i), knn, &QueryBitset);
        // std::cout << " topk: ";
        for (size_t j = 0; j < knn; j++) {
            if (!results.empty()) {
                ids.mutable_data(i)[j] = results.top().second;
                // std::cout << results.top().second << ' ';
                results.pop();
            } else {
                ids.mutable_data(i)[j] = 0;
            }
        }
        // std::cout << " done" << std::endl;
    }); 
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Time serve queries: " << std::chrono::duration<double>(end - start).count() << std::endl << std::flush;
    std::cout << "Num distance comps: " << _hnsw->metric_distance_computations << std::endl << std::flush;
    _hnsw->metric_distance_computations = 0;

    return std::move(ids);
  }
};

class OraclePartitionUint8 {
 public:
    static const int ser_version = 1;  // serialization version

    uint8_t* _data;
    hnswlib::PartitionedHNSW<int, uint8_t>* alg;
    hnswlib::DatasetFilters* dataset_filters;

    OraclePartitionUint8(
        std::string filename,
        std::string filter_filename,
        const std::vector<hnswlib::QueryFilter>& historical_workload,
        size_t dataset_size,
        size_t dim,
        size_t M,
        size_t ef_construction,
        size_t bitvector_cutoff,
        bool enable_heterogeneous_indexing,
        size_t num_threads
    ) {
        auto start = std::chrono::high_resolution_clock::now();
        // setup index params
        hnswlib::PartitionedIndexParams index_params{
            dataset_size,
            dim,
            M,
            ef_construction,
            std::numeric_limits<int>::max(), // Infinite budget
            bitvector_cutoff,
            std::numeric_limits<int>::max(), // Use all historical queries for optimization
            enable_heterogeneous_indexing,
            num_threads};

        // setup filters
        dataset_filters = new hnswlib::DatasetFilters(fopen(filter_filename.c_str(), "rb"), num_threads);
        dataset_filters->transpose_inplace();
        dataset_filters->make_bvs();

        // setup data
        std::ifstream reader(filename);
        assert(reader.is_open());
        size_t num_points;
        size_t d;
        reader.read((char*)(&num_points), sizeof(unsigned int));
        reader.read((char*)(&d), sizeof(unsigned int));

        _data = new uint8_t[dim * dataset_size];
        reader.read((char*)_data, sizeof(uint8_t) * dim * dataset_size);

        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Time to read data: " << std::chrono::duration<double>(end - start).count() << std::endl << std::flush;        

        hnswlib::L2SpaceX* space = new hnswlib::L2SpaceX(dim / 4);
        alg = new hnswlib::PartitionedHNSW<int, uint8_t>(_data, space, dataset_filters, index_params, historical_workload);
    }

    void update_index(const std::vector<hnswlib::QueryFilter>& historical_workload) {
        alg->updateIndexWorkload(historical_workload);
        alg->fitIndex();
    }

    py::array_t<unsigned int> batch_filter_search(
     py::array_t<uint8_t, py::array::c_style | py::array::forcecast>& queries,
     const std::vector<hnswlib::QueryFilter>& filters, uint64_t num_queries,
     uint64_t knn, size_t ef_search, uint64_t num_threads) {
    py::array_t<unsigned int> ids({num_queries, knn});

    
    auto start = std::chrono::high_resolution_clock::now();
    // Set ef_search of partitions
    alg->setEf(ef_search);
    hnswlib::Predicate predicate_arr[filters.size()];
    hnswlib::ParallelFor(0, filters.size(), num_threads, [&](size_t i, size_t threadId) {
        predicate_arr[i] = hnswlib::Predicate(dataset_filters, filters[i]);
    }); 
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Time construct predicates: " << std::chrono::duration<double>(end - start).count() << std::endl << std::flush;
    start = std::chrono::high_resolution_clock::now();
    hnswlib::ParallelFor(0, filters.size(), num_threads, [&](size_t i, size_t threadId) {
        auto results = alg->searchKnn(queries.data(i), knn, predicate_arr[i]);
        for (size_t j = 0; j < knn; j++) {
            ids.mutable_data(i)[j] = results.top().second;
            results.pop();
        }
    }); 
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Time serve queries: " << std::chrono::duration<double>(end - start).count() << std::endl << std::flush;
    alg->printTally();

    return ids;
  }
};

class OraclePartitionFloat {
 public:
    static const int ser_version = 1;  // serialization version

    float* _data;
    hnswlib::PartitionedHNSW<float, float>* alg;
    hnswlib::DatasetFilters* dataset_filters;

    OraclePartitionFloat(
        std::string filename,
        std::string filter_filename,
        const std::vector<hnswlib::QueryFilter>& historical_workload,
        size_t dataset_size,
        size_t dim,
        size_t M,
        size_t ef_construction,
        size_t bitvector_cutoff,
        bool enable_heterogeneous_indexing,
        size_t num_threads
    ) {
        auto start = std::chrono::high_resolution_clock::now();
        // setup index params
        hnswlib::PartitionedIndexParams index_params{
            dataset_size,
            dim,
            M,
            ef_construction,
            std::numeric_limits<int>::max(), // Infinite budget
            bitvector_cutoff,
            std::numeric_limits<int>::max(), // Use all historical queries for optimization
            enable_heterogeneous_indexing,
            num_threads};

        // setup filters
        dataset_filters = new hnswlib::DatasetFilters(fopen(filter_filename.c_str(), "rb"), num_threads);
        dataset_filters->transpose_inplace();
        dataset_filters->make_bvs();

        // setup data
        std::ifstream reader(filename);
        assert(reader.is_open());
        size_t num_points;
        size_t d;
        reader.read((char*)(&num_points), sizeof(unsigned int));
        reader.read((char*)(&d), sizeof(unsigned int));

        _data = new float[dim * dataset_size];
        reader.read((char*)_data, sizeof(float) * dim * dataset_size);

        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Time to read data: " << std::chrono::duration<double>(end - start).count() << std::endl << std::flush;        

        hnswlib::L2Space* space = new hnswlib::L2Space(dim);
        alg = new hnswlib::PartitionedHNSW<float, float>(_data, space, dataset_filters, index_params, historical_workload);
    }

    void update_index(const std::vector<hnswlib::QueryFilter>& historical_workload) {
        alg->updateIndexWorkload(historical_workload);
        alg->fitIndex();
    }

    py::array_t<unsigned int> batch_filter_search(
     py::array_t<float, py::array::c_style | py::array::forcecast>& queries,
     const std::vector<hnswlib::QueryFilter>& filters, uint64_t num_queries,
     uint64_t knn, size_t ef_search, uint64_t num_threads) {
    py::array_t<unsigned int> ids({num_queries, knn});

    
    auto start = std::chrono::high_resolution_clock::now();
    // Set ef_search of partitions
    alg->setEf(ef_search);
    hnswlib::Predicate predicate_arr[filters.size()];
    hnswlib::ParallelFor(0, filters.size(), num_threads, [&](size_t i, size_t threadId) {
        predicate_arr[i] = hnswlib::Predicate(dataset_filters, filters[i]);
    }); 
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Time construct predicates: " << std::chrono::duration<double>(end - start).count() << std::endl << std::flush;
    start = std::chrono::high_resolution_clock::now();
    hnswlib::ParallelFor(0, filters.size(), num_threads, [&](size_t i, size_t threadId) {
        auto results = alg->searchKnn(queries.data(i), knn, predicate_arr[i]);
        for (size_t j = 0; j < knn; j++) {
            ids.mutable_data(i)[j] = results.top().second;
            results.pop();
        }
    }); 
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Time serve queries: " << std::chrono::duration<double>(end - start).count() << std::endl << std::flush;
    alg->printTally();

    return ids;
  }
};

class TagTreePartitionUint8 {
 public:
    static const int ser_version = 1;  // serialization version

    uint8_t* _data;
    hnswlib::PartitionedHNSW<int, uint8_t>* alg;
    hnswlib::DatasetFilters* dataset_filters;

    TagTreePartitionUint8(
        std::string filename,
        std::string filter_filename,
        size_t dataset_size,
        size_t dim,
        size_t M,
        size_t ef_construction,
        size_t bitvector_cutoff,
        bool enable_heterogeneous_indexing,
        size_t num_threads
    ) {
        auto start = std::chrono::high_resolution_clock::now();
        // setup index params
        hnswlib::PartitionedIndexParams index_params{
            dataset_size,
            dim,
            M,
            ef_construction,
            std::numeric_limits<int>::max(), // Infinite budget
            bitvector_cutoff,
            std::numeric_limits<int>::max(), // Use all historical queries for optimization
            enable_heterogeneous_indexing,
            num_threads};

        // setup filters
        dataset_filters = new hnswlib::DatasetFilters(fopen(filter_filename.c_str(), "rb"), num_threads);

        // setup data
        std::ifstream reader(filename);
        assert(reader.is_open());
        size_t num_points;
        size_t d;
        reader.read((char*)(&num_points), sizeof(unsigned int));
        reader.read((char*)(&d), sizeof(unsigned int));

        _data = new uint8_t[dim * dataset_size];
        reader.read((char*)_data, sizeof(uint8_t) * dim * dataset_size);

        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Time to read data: " << std::chrono::duration<double>(end - start).count() << std::endl << std::flush;        

        // Setup all 1-tag filters as historical queries
        start = std::chrono::high_resolution_clock::now();
        std::vector<hnswlib::QueryFilter> tagtree_workload;
        for (int i = 0; i < dataset_filters->n_points; i++) {
            tagtree_workload.push_back(hnswlib::QueryFilter(std::unordered_set<int32_t>({i}), true));
        }
        end = std::chrono::high_resolution_clock::now();
        std::cout << "Time to setup tagtree predicates: " << std::chrono::duration<double>(end - start).count() << std::endl << std::flush; 
        std::cout << "Number of tagtree predicates: " << tagtree_workload.size() << std::endl << std::flush;  

        hnswlib::L2SpaceX* space = new hnswlib::L2SpaceX(dim / 4);
        alg = new hnswlib::PartitionedHNSW<int, uint8_t>(_data, space, dataset_filters, index_params, tagtree_workload);
    }

    void update_index(const std::vector<hnswlib::QueryFilter>& historical_workload) {
        alg->updateIndexWorkload(historical_workload);
        alg->fitIndex();
    }

    py::array_t<unsigned int> batch_filter_search(
     py::array_t<uint8_t, py::array::c_style | py::array::forcecast>& queries,
     const std::vector<hnswlib::QueryFilter>& filters, uint64_t num_queries,
     uint64_t knn, size_t ef_search, uint64_t num_threads) {
    py::array_t<unsigned int> ids({num_queries, knn});

    
    auto start = std::chrono::high_resolution_clock::now();
    // Set ef_search of partitions
    alg->setEf(ef_search);
    hnswlib::Predicate predicate_arr[filters.size()];
    hnswlib::ParallelFor(0, filters.size(), num_threads, [&](size_t i, size_t threadId) {
        predicate_arr[i] = hnswlib::Predicate(dataset_filters, filters[i]);
    }); 
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Time construct predicates: " << std::chrono::duration<double>(end - start).count() << std::endl << std::flush;
    start = std::chrono::high_resolution_clock::now();
    hnswlib::ParallelFor(0, filters.size(), num_threads, [&](size_t i, size_t threadId) {
        auto results = alg->searchKnn(queries.data(i), knn, predicate_arr[i]);
        for (size_t j = 0; j < knn; j++) {
            ids.mutable_data(i)[j] = results.top().second;
            results.pop();
        }
    }); 
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Time serve queries: " << std::chrono::duration<double>(end - start).count() << std::endl << std::flush;
    alg->printTally();

    return ids;
  }
};

class TagTreePartitionFloat {
 public:
    static const int ser_version = 1;  // serialization version

    float* _data;
    hnswlib::PartitionedHNSW<float, float>* alg;
    hnswlib::DatasetFilters* dataset_filters;

    TagTreePartitionFloat(
        std::string filename,
        std::string filter_filename,
        size_t dataset_size,
        size_t dim,
        size_t M,
        size_t ef_construction,
        size_t bitvector_cutoff,
        bool enable_heterogeneous_indexing,
        size_t num_threads
    ) {
        auto start = std::chrono::high_resolution_clock::now();
        // setup index params
        hnswlib::PartitionedIndexParams index_params{
            dataset_size,
            dim,
            M,
            ef_construction,
            std::numeric_limits<int>::max(), // Infinite budget
            bitvector_cutoff,
            std::numeric_limits<int>::max(), // Use all historical queries for optimization
            enable_heterogeneous_indexing,
            num_threads};

        // setup filters
        dataset_filters = new hnswlib::DatasetFilters(fopen(filter_filename.c_str(), "rb"), num_threads);

        // setup data
        std::ifstream reader(filename);
        assert(reader.is_open());
        size_t num_points;
        size_t d;
        reader.read((char*)(&num_points), sizeof(unsigned int));
        reader.read((char*)(&d), sizeof(unsigned int));

        _data = new float[dim * dataset_size];
        reader.read((char*)_data, sizeof(float) * dim * dataset_size);

        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Time to read data: " << std::chrono::duration<double>(end - start).count() << std::endl << std::flush;        

        // Setup all 1-tag filters as historical queries
        start = std::chrono::high_resolution_clock::now();
        std::vector<hnswlib::QueryFilter> tagtree_workload;
        for (int i = 0; i < dataset_filters->n_points; i++) {
            tagtree_workload.push_back(hnswlib::QueryFilter(std::unordered_set<int32_t>({i}), true));
        }
        end = std::chrono::high_resolution_clock::now();
        std::cout << "Time to setup tagtree predicates: " << std::chrono::duration<double>(end - start).count() << std::endl << std::flush; 
        std::cout << "Number of tagtree predicates: " << tagtree_workload.size() << std::endl << std::flush;  

        hnswlib::L2Space* space = new hnswlib::L2Space(dim);
        alg = new hnswlib::PartitionedHNSW<float, float>(_data, space, dataset_filters, index_params, tagtree_workload);
    }

    void update_index(const std::vector<hnswlib::QueryFilter>& historical_workload) {
        alg->updateIndexWorkload(historical_workload);
        alg->fitIndex();
    }

    py::array_t<unsigned int> batch_filter_search(
     py::array_t<float, py::array::c_style | py::array::forcecast>& queries,
     const std::vector<hnswlib::QueryFilter>& filters, uint64_t num_queries,
     uint64_t knn, size_t ef_search, uint64_t num_threads) {
    py::array_t<unsigned int> ids({num_queries, knn});

    
    auto start = std::chrono::high_resolution_clock::now();
    // Set ef_search of partitions
    alg->setEf(ef_search);
    hnswlib::Predicate predicate_arr[filters.size()];
    hnswlib::ParallelFor(0, filters.size(), num_threads, [&](size_t i, size_t threadId) {
        predicate_arr[i] = hnswlib::Predicate(dataset_filters, filters[i]);
    }); 
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Time construct predicates: " << std::chrono::duration<double>(end - start).count() << std::endl << std::flush;
    start = std::chrono::high_resolution_clock::now();
    hnswlib::ParallelFor(0, filters.size(), num_threads, [&](size_t i, size_t threadId) {
        auto results = alg->searchKnn(queries.data(i), knn, predicate_arr[i]);
        for (size_t j = 0; j < knn; j++) {
            ids.mutable_data(i)[j] = results.top().second;
            results.pop();
        }
    }); 
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Time serve queries: " << std::chrono::duration<double>(end - start).count() << std::endl << std::flush;
    alg->printTally();

    return ids;
  }
};

class AcornIndexUint8 {
 public:
    static const int ser_version = 1;  // serialization version

    uint8_t* _data;
    float* _float_data;
    faiss::IndexACORNFlat* acorn_gamma;
    hnswlib::DatasetFilters* dataset_filters;
    size_t _dataset_size;
    size_t _dim;
    float _bruteforce_selectivity_threshold;
    hnswlib::HierarchicalNSW<int>* _hnsw;
    bool init_bvs = false;

    AcornIndexUint8(
        std::string filename,
        std::string filter_filename,
        size_t dataset_size,
        size_t dim,
        size_t M,
        size_t gamma,
        size_t m_beta,
        float bruteforce_selectivity_threshold,
        size_t num_threads
    ) : _dataset_size(dataset_size), _dim(dim), _bruteforce_selectivity_threshold(bruteforce_selectivity_threshold) {
        auto start = std::chrono::high_resolution_clock::now();
        // setup filters
        std::cout << "Start read filters" << std::endl << std::flush;
        dataset_filters = new hnswlib::DatasetFilters(fopen(filter_filename.c_str(), "rb"), num_threads);
        dataset_filters->transpose_inplace();
        dataset_filters->make_bvs();
        std::cout << "Read filters" << std::endl << std::flush;

        // setup data
        std::ifstream reader(filename);
        assert(reader.is_open());
        size_t num_points;
        size_t d;
        reader.read((char*)(&num_points), sizeof(unsigned int));
        reader.read((char*)(&d), sizeof(unsigned int));
        std::cout << "Num points:" << num_points << std::endl << std::flush;
        std::cout << "d:" << d << std::endl << std::flush;

        _data = new uint8_t[dim * dataset_size];
        reader.read((char*)_data, sizeof(uint8_t) * dim * dataset_size);
        std::cout << "Read data" << std::endl << std::flush;

        _float_data = new float[dim * dataset_size];
        for (int i = 0; i < dim * dataset_size; ++i) {
            _float_data[i] = (float)_data[i];
        }
        std::cout << "Converted data" << std::endl << std::flush;

        std::vector<int> metadata;
        for (size_t i = 0; i < dataset_size; i++) {
            metadata.push_back(0);
        }

        //Just here for the distance calcs
        hnswlib::L2SpaceX* space = new hnswlib::L2SpaceX(dim / 4);
        _hnsw = new hnswlib::HierarchicalNSW<int>(space, 10, 16, 40);

        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Time to read data: " << std::chrono::duration<double>(end - start).count() << std::endl << std::flush;

        start = std::chrono::high_resolution_clock::now();
        // Create acorn index
        omp_set_num_threads(num_threads);
        acorn_gamma = new faiss::IndexACORNFlat(dim, M, gamma, metadata, m_beta);

        // Add vectors
        acorn_gamma->add(dataset_size, _float_data);
        end = std::chrono::high_resolution_clock::now();
        std::cout << "Time to build index: " << std::chrono::duration<double>(end - start).count() << std::endl << std::flush;
    }

        py::array_t<unsigned int> batch_filter_search(
     py::array_t<uint8_t, py::array::c_style | py::array::forcecast>& queries,
     const std::vector<hnswlib::QueryFilter>& filters, uint64_t num_queries,
     uint64_t knn, size_t ef_search, uint64_t num_threads) {
    omp_set_num_threads(num_threads);
    py::array_t<unsigned int> ids({num_queries, knn});
    acorn_gamma->acorn.efSearch = ef_search;
    
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<hnswlib::Predicate> predicate_arr;
    for (int i = 0; i < filters.size(); i++) {
        predicate_arr.push_back(hnswlib::Predicate(dataset_filters, filters[i]));
    } 

    float acorn_search_time = 0;
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Time construct predicates: " << std::chrono::duration<double>(end - start).count() << std::endl << std::flush;
    start = std::chrono::high_resolution_clock::now();

    float bruteforce_search_time = 0;
    size_t bruteforce_distance_comps = 0;
    size_t bruteforce_searches = 0;
    size_t new_dim = _dim / 4;
    hnswlib::ParallelFor(0, filters.size(), num_threads, [&](size_t i, size_t threadId) {
        std::vector<faiss::idx_t> nns2(knn);
        std::vector<float> dis2(knn);
        std::vector<char> filter_ids_map(_dataset_size);
        // Bruteforce search
        if ((float) predicate_arr[i].cardinality() / _dataset_size < _bruteforce_selectivity_threshold) {
            auto start = std::chrono::high_resolution_clock::now();
            roaring::Roaring& deref = *predicate_arr[i]._bitvector;
            std::priority_queue<std::pair<int, size_t>> max_priority_queue;
            for (roaring::Roaring::const_iterator j = deref.begin(); j != deref.end(); j++) {
                //int dist = hnswlib::full_distance_avx2(queries.data(i), (_data + _dim * *j), &new_dim);
                int dist = _hnsw->fstdistfunc_(queries.data(i), (_data + _dim * *j), _hnsw->dist_func_param_);
                if (max_priority_queue.size() < knn) {
                    max_priority_queue.push(std::make_pair(dist, *j));
                } else if (dist < max_priority_queue.top().first) {
                    max_priority_queue.pop();
                    max_priority_queue.push(std::make_pair(dist, *j));
                }
            }
            bruteforce_distance_comps += predicate_arr[i].cardinality();
            for (size_t j = 0; j < knn; j++) {
                ids.mutable_data(i)[j] = max_priority_queue.top().second;
                max_priority_queue.pop();
            }
            auto end = std::chrono::high_resolution_clock::now();
            bruteforce_search_time += std::chrono::duration<double>(end - start).count();
            bruteforce_searches++;
            bruteforce_distance_comps += predicate_arr[i].cardinality();
        } else {
            std::vector<faiss::idx_t> nns2(knn);
            std::vector<float> dis2(knn);
            std::vector<char> filter_ids_map(_dataset_size);
            for (roaring::Roaring::const_iterator j = predicate_arr[i]._bitvector->begin(); j != predicate_arr[i]._bitvector->end(); j++) {
                filter_ids_map[*j] = (bool) 1;
            }
            // std::vector<uint32_t> tmp_vec = predicate_arr[i].matching_points();
            // for (size_t j = 0; j < tmp_vec.size(); j++) {
            //     filter_ids_map[tmp_vec[j]] = (bool) 1;
            // }
            float* float_query = new float[_dim];
            for (size_t j = 0; j < _dim; j++) {
                float_query[j] = static_cast<float>(queries.data(i)[j]);
            }
            auto start2 = std::chrono::high_resolution_clock::now();
            acorn_gamma->search(1, float_query, knn, dis2.data(), nns2.data(), filter_ids_map.data());
            for (size_t j = 0; j < knn; j++) {
                ids.mutable_data(i)[j] = nns2[j];
            }
            auto end2 = std::chrono::high_resolution_clock::now();
            acorn_search_time += std::chrono::duration<double>(end2 - start2).count();
        }
    }); 
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Time spent by acorn search: " << acorn_search_time << std::endl << std::flush;
    const faiss::ACORNStats& stats = faiss::acorn_stats;
    std::cout << "Acorn distance comps: " << stats.n3 << std::endl;
    std::cout << "Time spent by bruteforce search: " << bruteforce_search_time << std::endl;
    std::cout << "Bruteforce distance comps: " << bruteforce_distance_comps << std::endl;
    std::cout << "Bruteforce searches: " << bruteforce_searches << std::endl;
    std::cout << "Time serve queries: " << std::chrono::duration<double>(end - start).count() << std::endl << std::flush;

    return std::move(ids);
  }
};


class AcornIndexFloat {
 public:
    static const int ser_version = 1;  // serialization version

    float* _data;
    faiss::IndexACORNFlat* acorn_gamma;
    hnswlib::DatasetFilters* dataset_filters;
    size_t _dataset_size;
    size_t _dim;
    float _bruteforce_selectivity_threshold;
    hnswlib::HierarchicalNSW<float>* _hnsw;
    bool init_bvs = false;

    AcornIndexFloat(
        std::string filename,
        std::string filter_filename,
        size_t dataset_size,
        size_t dim,
        size_t M,
        size_t gamma,
        size_t m_beta,
        float bruteforce_selectivity_threshold,
        size_t num_threads
    ) : _dataset_size(dataset_size), _dim(dim), _bruteforce_selectivity_threshold(bruteforce_selectivity_threshold) {
        auto start = std::chrono::high_resolution_clock::now();
        // setup filters
        std::cout << "Start read filters" << std::endl << std::flush;
        dataset_filters = new hnswlib::DatasetFilters(fopen(filter_filename.c_str(), "rb"), num_threads);
        dataset_filters->transpose_inplace();
        dataset_filters->make_bvs();
        std::cout << "Read filters" << std::endl << std::flush;

        // setup data
        std::ifstream reader(filename);
        assert(reader.is_open());
        size_t num_points;
        size_t d;
        reader.read((char*)(&num_points), sizeof(unsigned int));
        reader.read((char*)(&d), sizeof(unsigned int));
        std::cout << "Num points:" << num_points << std::endl << std::flush;
        std::cout << "d:" << d << std::endl << std::flush;

        _data = new float[dim * dataset_size];
        reader.read((char*)_data, sizeof(float) * dim * dataset_size);
        std::cout << "Read data" << std::endl << std::flush;

        std::vector<int> metadata;
        for (size_t i = 0; i < dataset_size; i++) {
            metadata.push_back(0);
        }

        //Just here for the distance calcs
        hnswlib::L2Space* space = new hnswlib::L2Space(dim);
        _hnsw = new hnswlib::HierarchicalNSW<float>(space, 10, 16, 40);

        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Time to read data: " << std::chrono::duration<double>(end - start).count() << std::endl << std::flush;

        start = std::chrono::high_resolution_clock::now();
        // Create acorn index
        omp_set_num_threads(num_threads);
        acorn_gamma = new faiss::IndexACORNFlat(dim, M, gamma, metadata, m_beta);

        // Add vectors
        acorn_gamma->add(dataset_size, _data);
        end = std::chrono::high_resolution_clock::now();
        std::cout << "Time to build index: " << std::chrono::duration<double>(end - start).count() << std::endl << std::flush;
    }

    py::array_t<unsigned int> batch_filter_search(
     py::array_t<float, py::array::c_style | py::array::forcecast>& queries,
     const std::vector<hnswlib::QueryFilter>& filters, uint64_t num_queries,
     uint64_t knn, size_t ef_search, uint64_t num_threads) {
    py::array_t<unsigned int> ids({num_queries, knn});
    omp_set_num_threads(num_threads);
    acorn_gamma->acorn.efSearch = ef_search;
    
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<hnswlib::Predicate> predicate_arr;
    for (int i = 0; i < filters.size(); i++) {
        predicate_arr.push_back(hnswlib::Predicate(dataset_filters, filters[i]));
    } 

    float acorn_search_time = 0;
    float maxsel_search_time = 0;
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Time construct predicates: " << std::chrono::duration<double>(end - start).count() << std::endl << std::flush;
    start = std::chrono::high_resolution_clock::now();
    float bruteforce_search_time = 0;
    size_t bruteforce_distance_comps = 0;
    size_t bruteforce_searches = 0;
    
    hnswlib::ParallelFor(0, filters.size(), num_threads, [&](size_t i, size_t threadId) {
        std::vector<faiss::idx_t> nns2(knn);
        std::vector<float> dis2(knn);
        std::vector<char> filter_ids_map(_dataset_size);

        // Bruteforce search
        if ((float) predicate_arr[i].cardinality() / _dataset_size <= _bruteforce_selectivity_threshold) {
            auto start = std::chrono::high_resolution_clock::now();
            roaring::Roaring& deref = *predicate_arr[i]._bitvector;
            std::priority_queue<std::pair<float, size_t>> max_priority_queue;
            for (roaring::Roaring::const_iterator j = deref.begin(); j != deref.end(); j++) {
                float dist = Faiss_fvec_L2sqr(queries.data(i), (_data + _dim * *j), _dim);
                if (max_priority_queue.size() < knn) {
                    max_priority_queue.push(std::make_pair(dist, *j));
                } else if (dist < max_priority_queue.top().first) {
                    max_priority_queue.pop();
                    max_priority_queue.push(std::make_pair(dist, *j));
                }
            }
            bruteforce_distance_comps += predicate_arr[i].cardinality();
            bruteforce_searches++;
            for (size_t j = 0; j < knn; j++) {
                ids.mutable_data(i)[j] = max_priority_queue.top().second;
                max_priority_queue.pop();
            }
            auto end = std::chrono::high_resolution_clock::now();
            bruteforce_search_time += std::chrono::duration<double>(end - start).count();
        } else {
            std::vector<uint32_t> tmp_vec = predicate_arr[i].matching_points();
            for (size_t j = 0; j < tmp_vec.size(); j++) {
                filter_ids_map[tmp_vec[j]] = (bool) 1;
            }
            auto start2 = std::chrono::high_resolution_clock::now();
            acorn_gamma->search(1, queries.data(i), knn, dis2.data(), nns2.data(), filter_ids_map.data());
            auto end2 = std::chrono::high_resolution_clock::now();
            for (size_t j = 0; j < knn; j++) {
                ids.mutable_data(i)[j] = nns2[j];
            }
            acorn_search_time += std::chrono::duration<double>(end2 - start2).count();
            // if (predicate_arr[i].cardinality() == _dataset_size) {
            //     maxsel_search_time += std::chrono::duration<double>(end2 - start2).count();
            // }
        }
    }); 
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Time spent by acorn search: " << acorn_search_time << std::endl << std::flush;
    const faiss::ACORNStats& stats = faiss::acorn_stats;
    std::cout << "Acorn distance comps: " << stats.n1 + stats.n2 + stats.n3 << std::endl;
    std::cout << "Time spent by bruteforce search: " << bruteforce_search_time << std::endl;
    std::cout << "Bruteforce distance comps: " << bruteforce_distance_comps << std::endl;
    std::cout << "Bruteforce searches: " << bruteforce_searches << std::endl;
    std::cout << "Time serve queries: " << std::chrono::duration<double>(end - start).count() << std::endl << std::flush;
    std::cout << "max selectivity query time: " << maxsel_search_time << std::endl << std::flush;

    return std::move(ids);
  }
};

class Caps {
 public:
    static const int ser_version = 1;  // serialization version

    float* _data;
    hnswlib::DatasetFilters* dataset_filters;
    FilterIndex* _caps_index;
    size_t _dim;

    Caps(
        std::string filename,
        std::string filter_filename,
        size_t dataset_size,
        size_t dim,
        size_t num_clusters,
        size_t num_threads
    ) : _dim(dim) {
        auto start = std::chrono::high_resolution_clock::now();
        // setup filters
        dataset_filters = new hnswlib::DatasetFilters(fopen(filter_filename.c_str(), "rb"), num_threads);

        // setup data
        std::ifstream reader(filename);
        assert(reader.is_open());
        size_t num_points;
        size_t d;
        reader.read((char*)(&num_points), sizeof(unsigned int));
        reader.read((char*)(&d), sizeof(unsigned int));

        _data = new float[dim * dataset_size];
        reader.read((char*)_data, sizeof(float) * dim * dataset_size);
        std::cout << "Read data" << std::endl << std::flush;
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Time to read data: " << std::chrono::duration<double>(end - start).count() << std::endl << std::flush;

        start = std::chrono::high_resolution_clock::now();
        vector<vector<string>> properties_arr;
        properties_arr.resize(dataset_size);
        hnswlib::ParallelFor(0, dataset_size, num_threads, [&](size_t i, size_t threadId) {
            vector<string> properties_vec;            
            auto tmp_vec = std::vector<uint32_t>(dataset_filters->row_indices.get() + dataset_filters->row_offsets[i], dataset_filters->row_indices.get() + dataset_filters->row_offsets[i + 1]);
            auto tmp_set = std::unordered_set<uint32_t>(tmp_vec.begin(), tmp_vec.end());
            for (uint32_t j = 0; j < dataset_filters->n_filters; j++) {
                if (tmp_set.find(j) != tmp_set.end()) {
                    properties_vec.push_back(std::to_string(j * 2 + 1));
                } else {
                    properties_vec.push_back(std::to_string(j * 2));
                }
            }
            properties_arr[i] = properties_vec;
        }); 
        end = std::chrono::high_resolution_clock::now();
        std::cout << "Time to setup properties: " << std::chrono::duration<double>(end - start).count() << std::endl << std::flush;

        dataset_filters->transpose_inplace();
        dataset_filters->make_bvs();
        
        _caps_index = new FilterIndex(_data, dim, dataset_size, num_clusters, properties_arr, "kmeans", 1);
        std::cout << "Finish init index " << std::endl << std::flush;
        _caps_index->get_index("L2", "/data/elastic-notebook/tmp/caps", 1);
        _caps_index->get_mc_propertiesIndex();
        _caps_index->loadIndex("/data/elastic-notebook/tmp/caps");
    }

    py::array_t<unsigned int> batch_filter_search(
     py::array_t<float, py::array::c_style | py::array::forcecast>& queries,
     const std::vector<hnswlib::QueryFilter>& filters, uint64_t num_queries,
     uint64_t knn, size_t num_to_check, uint64_t num_threads) {
    py::array_t<unsigned int> ids({num_queries, knn});

    auto start = std::chrono::high_resolution_clock::now();
    vector<vector<string>> query_attr_arr;
    query_attr_arr.resize(filters.size());
    hnswlib::ParallelFor(0, filters.size(), num_threads, [&](size_t i, size_t threadId) {
        vector<string> query_attr_vec;            
        auto tmp_set = std::unordered_set<uint32_t>(filters[i]._filters.begin(), filters[i]._filters.end());
        for (uint32_t j = 0; j < dataset_filters->n_points; j++) {
            if (tmp_set.find(j) != tmp_set.end()) {
                query_attr_vec.push_back(std::to_string(j * 2 + 1));
            } else {
                query_attr_vec.push_back("X");
            }
        }
        query_attr_arr[i] = query_attr_vec;
    }); 
        // sanity check
    for (auto i: query_attr_arr[0])
        std::cout << i << ' ';
    std::cout << std::endl;

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Time construct predicates: " << std::chrono::duration<double>(end - start).count() << std::endl << std::flush;

    start = std::chrono::high_resolution_clock::now();
    float* query_vecs = new float[filters.size() * _dim];
    for (size_t i = 0; i < filters.size(); i++) {
        for (size_t j = 0; j < _dim; j++) {
            query_vecs[i * _dim + j] = queries.data(i)[j];
        }
    }
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Time convert data: " << std::chrono::duration<double>(end - start).count() << std::endl << std::flush;

    start = std::chrono::high_resolution_clock::now();
    _caps_index->query(query_vecs, filters.size(), query_attr_arr, knn, num_to_check);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Time serve queries: " << std::chrono::duration<double>(end - start).count() << std::endl << std::flush;

    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < filters.size(); i++) {
        for (size_t j = 0; j < knn; j++) {
            ids.mutable_data(i)[j] = _caps_index->neighbor_set[i * knn + j];
        }
    }
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Time set results: " << std::chrono::duration<double>(end - start).count() << std::endl << std::flush;
    std::cout << "Num distance comps: " << _caps_index->num_dists << std::endl << std::flush;
    _caps_index->num_dists = 0;

    return ids;
  }
};


class NHQ {
 public:
    static const int ser_version = 1;  // serialization version

    float* _data;
    hnswlib::DatasetFilters* dataset_filters;
    efanna2e::IndexGraph* _nhq_index;
    size_t _dim;

    NHQ(
        std::string filename,
        std::string filter_filename,
        size_t dataset_size,
        size_t dim,
        size_t K,
        size_t L,
        size_t iter,
        size_t S,
        size_t R,
        size_t RANGE,
        size_t PL,
        float B,
        float M,
        size_t num_threads
    ) : _dim(dim) {
        auto start = std::chrono::high_resolution_clock::now();
        // setup filters
        dataset_filters = new hnswlib::DatasetFilters(fopen(filter_filename.c_str(), "rb"), num_threads);

        // setup data
        std::ifstream reader(filename);
        assert(reader.is_open());
        size_t num_points;
        size_t d;
        reader.read((char*)(&num_points), sizeof(unsigned int));
        reader.read((char*)(&d), sizeof(unsigned int));

        _data = new float[dim * dataset_size];
        reader.read((char*)_data, sizeof(float) * dim * dataset_size);
        std::cout << "Read data" << std::endl << std::flush;
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Time to read data: " << std::chrono::duration<double>(end - start).count() << std::endl << std::flush;

        start = std::chrono::high_resolution_clock::now();
        vector<vector<string>> properties_arr;
        properties_arr.resize(dataset_size);
        hnswlib::ParallelFor(0, dataset_size, num_threads, [&](size_t i, size_t threadId) {
            vector<string> properties_vec;            
            auto tmp_vec = std::vector<uint32_t>(dataset_filters->row_indices.get() + dataset_filters->row_offsets[i], dataset_filters->row_indices.get() + dataset_filters->row_offsets[i + 1]);
            auto tmp_set = std::unordered_set<uint32_t>(tmp_vec.begin(), tmp_vec.end());
            for (uint32_t j = 0; j < dataset_filters->n_filters; j++) {
                if (tmp_set.find(j) != tmp_set.end()) {
                    properties_vec.push_back(std::to_string(j * 2 + 1));
                } else {
                    properties_vec.push_back(std::to_string(j * 2));
                }
            }
            properties_arr[i] = properties_vec;
        }); 
        end = std::chrono::high_resolution_clock::now();
        std::cout << "Time to setup properties: " << std::chrono::duration<double>(end - start).count() << std::endl << std::flush;

        dataset_filters->transpose_inplace();
        dataset_filters->make_bvs();

        efanna2e::IndexRandom init_index(dim, dataset_size);
        _nhq_index = new efanna2e::IndexGraph(dim, dataset_size, efanna2e::L2, (efanna2e::Index *)(&init_index));
        for (int i = 0; i < dataset_size; i++)
        {
          _nhq_index->AddAllNodeAttributes(properties_arr[i]);
        }

        efanna2e::Parameters paras;
        paras.Set<unsigned>("K", K);
        paras.Set<unsigned>("L", L);
        paras.Set<unsigned>("iter", iter);
        paras.Set<unsigned>("S", S);
        paras.Set<unsigned>("R", R);
        paras.Set<unsigned>("RANGE", RANGE);
        paras.Set<unsigned>("PL", PL);
        paras.Set<float>("B", B);
        paras.Set<float>("M", M);
        unsigned ddim = dim;
        _data = efanna2e::data_align(_data, dataset_size, ddim);
        _nhq_index->Build(dataset_size, _data, paras);
        _nhq_index->OptimizeGraph(_data);
        std::cout << "Finish init index " << std::endl << std::flush;
    }

    py::array_t<unsigned int> batch_filter_search(
     py::array_t<float, py::array::c_style | py::array::forcecast>& queries,
     const std::vector<hnswlib::QueryFilter>& filters, uint64_t num_queries,
     uint64_t knn, size_t L_search, float weight_search, uint64_t num_threads) {
    py::array_t<unsigned int> ids({num_queries, knn});
    
    auto start = std::chrono::high_resolution_clock::now();
    vector<vector<string>> query_attr_arr;
    query_attr_arr.resize(filters.size());
    hnswlib::ParallelFor(0, filters.size(), num_threads, [&](size_t i, size_t threadId) {
        vector<string> query_attr_vec;            
        auto tmp_set = std::unordered_set<uint32_t>(filters[i]._filters.begin(), filters[i]._filters.end());
        for (uint32_t j = 0; j < dataset_filters->n_points; j++) {
            if (tmp_set.find(j) != tmp_set.end()) {
                query_attr_vec.push_back(std::to_string(j * 2 + 1));
            } else {
                query_attr_vec.push_back(std::to_string(j * 2));
            }
        }
        query_attr_arr[i] = query_attr_vec;
    }); 
        // sanity check
    for (auto i: query_attr_arr[0])
        std::cout << i << ' ';
    std::cout << std::endl;

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Time construct predicates: " << std::chrono::duration<double>(end - start).count() << std::endl << std::flush;

    start = std::chrono::high_resolution_clock::now();
    float* query_vecs = new float[filters.size() * _dim];
    for (size_t i = 0; i < filters.size(); i++) {
        for (size_t j = 0; j < _dim; j++) {
            query_vecs[i * _dim + j] = queries.data(i)[j];
        }
    }
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Time convert data: " << std::chrono::duration<double>(end - start).count() << std::endl << std::flush;

    start = std::chrono::high_resolution_clock::now();
    efanna2e::Parameters paras;
    paras.Set<unsigned>("L_search", L_search);
    paras.Set<float>("weight_search", weight_search);
    hnswlib::ParallelFor(0, filters.size(), num_threads, [&](size_t i, size_t threadId) {
        _nhq_index->SearchWithOptGraph(query_attr_arr[i], queries.data(i), knn, paras, ids.mutable_data(i));
    }); 
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Time serve queries: " << std::chrono::duration<double>(end - start).count() << std::endl << std::flush;

    return ids;
  }
};



PYBIND11_MODULE(hnswlib, m) {
        // should have initializers taking either one or two int32_t arguments
        py::class_<hnswlib::QueryFilter>(m, "QueryFilter")
        .def(py::init<std::unordered_set<int32_t>, bool>(), py::arg("filters"), py::arg("is_and"));

        py::class_<HierarchicalIndexUint8>(m, "HierarchicalIndexUint8")
        .def(py::init<std::string, std::string, std::vector<hnswlib::QueryFilter>&, size_t, size_t, size_t, size_t, size_t, size_t, size_t, bool, size_t>(),
            py::arg("filename"), py::arg("filter_filename"), py::arg("historical_workload"), py::arg("dataset_size"), py::arg("dim"), py::arg("M"), py::arg("ef_construction"), py::arg("index_vector_budget"), py::arg("bitvector_cutoff"), py::arg("historical_workload_window_size"), py::arg("enable_heterogeneous_indexing"), py::arg("num_threads"))
        .def("update_index", &HierarchicalIndexUint8::update_index)
        .def("batch_filter_search", &HierarchicalIndexUint8::batch_filter_search);

        py::class_<HierarchicalIndexFloat>(m, "HierarchicalIndexFloat")
        .def(py::init<std::string, std::string, std::vector<hnswlib::QueryFilter>&, size_t, size_t, size_t, size_t, size_t, size_t, size_t, bool, size_t>(),
            py::arg("filename"), py::arg("filter_filename"), py::arg("historical_workload"), py::arg("dataset_size"), py::arg("dim"), py::arg("M"), py::arg("ef_construction"), py::arg("index_vector_budget"), py::arg("bitvector_cutoff"), py::arg("historical_workload_window_size"), py::arg("enable_heterogeneous_indexing"), py::arg("num_threads"))
        .def("update_index", &HierarchicalIndexFloat::update_index)
        .def("update_data", &HierarchicalIndexFloat::update_data)
        .def("batch_filter_search", &HierarchicalIndexFloat::batch_filter_search);

        py::class_<PreFilterUint8>(m, "PreFilterUint8")
        .def(py::init<std::string, std::string, size_t, size_t, size_t>(),
            py::arg("filename"), py::arg("filter_filename"), py::arg("dataset_size"), py::arg("dim"), py::arg("num_threads"))
        .def("batch_filter_search", &PreFilterUint8::batch_filter_search);

        py::class_<PreFilterFloat>(m, "PreFilterFloat")
        .def(py::init<std::string, std::string, size_t, size_t, size_t>(),
            py::arg("filename"), py::arg("filter_filename"), py::arg("dataset_size"), py::arg("dim"), py::arg("num_threads"))
        .def("batch_filter_search", &PreFilterFloat::batch_filter_search);

        py::class_<HNSWBaseUint8>(m, "HNSWBaseUint8")
        .def(py::init<std::string, std::string, size_t, size_t, size_t, size_t, size_t>(),
            py::arg("filename"), py::arg("filter_filename"), py::arg("dataset_size"), py::arg("dim"), py::arg("M"), py::arg("ef_construction"), py::arg("num_threads"))
        .def("batch_filter_search", &HNSWBaseUint8::batch_filter_search);

        py::class_<HNSWBaseFloat>(m, "HNSWBaseFloat")
        .def(py::init<std::string, std::string, size_t, size_t, size_t, size_t, size_t>(),
            py::arg("filename"), py::arg("filter_filename"), py::arg("dataset_size"), py::arg("dim"), py::arg("M"), py::arg("ef_construction"), py::arg("num_threads"))
        .def("batch_filter_search", &HNSWBaseFloat::batch_filter_search);

        py::class_<OraclePartitionUint8>(m, "OraclePartitionUint8")
        .def(py::init<std::string, std::string, std::vector<hnswlib::QueryFilter>&, size_t, size_t, size_t, size_t, size_t, bool, size_t>(),
            py::arg("filename"), py::arg("filter_filename"), py::arg("historical_workload"), py::arg("dataset_size"), py::arg("dim"), py::arg("M"), py::arg("ef_construction"), py::arg("bitvector_cutoff"), py::arg("enable_heterogeneous_indexing"), py::arg("num_threads"))
        .def("update_index", &OraclePartitionUint8::update_index)
        .def("batch_filter_search", &OraclePartitionUint8::batch_filter_search);

        py::class_<OraclePartitionFloat>(m, "OraclePartitionFloat")
        .def(py::init<std::string, std::string, std::vector<hnswlib::QueryFilter>&, size_t, size_t, size_t, size_t, size_t, bool, size_t>(),
            py::arg("filename"), py::arg("filter_filename"), py::arg("historical_workload"), py::arg("dataset_size"), py::arg("dim"), py::arg("M"), py::arg("ef_construction"), py::arg("bitvector_cutoff"), py::arg("enable_heterogeneous_indexing"), py::arg("num_threads"))
        .def("update_index", &OraclePartitionFloat::update_index)
        .def("batch_filter_search", &OraclePartitionFloat::batch_filter_search);

        py::class_<TagTreePartitionUint8>(m, "TagTreePartitionUint8")
        .def(py::init<std::string, std::string, size_t, size_t, size_t, size_t, size_t, bool, size_t>(),
            py::arg("filename"), py::arg("filter_filename"), py::arg("dataset_size"), py::arg("dim"), py::arg("M"), py::arg("ef_construction"), py::arg("bitvector_cutoff"), py::arg("enable_heterogeneous_indexing"), py::arg("num_threads"))
        .def("update_index", &TagTreePartitionUint8::update_index)
        .def("batch_filter_search", &TagTreePartitionUint8::batch_filter_search);

        py::class_<TagTreePartitionFloat>(m, "TagTreePartitionFloat")
        .def(py::init<std::string, std::string, size_t, size_t, size_t, size_t, size_t, bool, size_t>(),
            py::arg("filename"), py::arg("filter_filename"), py::arg("dataset_size"), py::arg("dim"), py::arg("M"), py::arg("ef_construction"), py::arg("bitvector_cutoff"), py::arg("enable_heterogeneous_indexing"), py::arg("num_threads"))
        .def("update_index", &TagTreePartitionFloat::update_index)
        .def("batch_filter_search", &TagTreePartitionFloat::batch_filter_search);

        py::class_<Caps>(m, "Caps")
        .def(py::init<std::string, std::string, size_t, size_t, size_t, size_t>(),
            py::arg("filename"), py::arg("filter_filename"), py::arg("dataset_size"), py::arg("dim"), py::arg("num_clusters"), py::arg("num_threads"))
        .def("batch_filter_search", &Caps::batch_filter_search);

        py::class_<NHQ>(m, "NHQ")
        .def(py::init<std::string, std::string, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, float, float, size_t>(),
            py::arg("filename"), py::arg("filter_filename"), py::arg("dataset_size"), py::arg("dim"), py::arg("K"), py::arg("L"), py::arg("iter"), py::arg("S"), py::arg("R"), py::arg("RANGE"), py::arg("PL"), py::arg("B"), py::arg("M"), py::arg("num_threads"))
        .def("batch_filter_search", &NHQ::batch_filter_search);

        py::class_<AcornIndexUint8>(m, "AcornIndexUint8")
        .def(py::init<std::string, std::string, size_t, size_t, size_t, size_t, size_t, float, size_t>(),
            py::arg("filename"), py::arg("filter_filename"), py::arg("dataset_size"), py::arg("dim"), py::arg("M"), py::arg("gamma"), py::arg("m_beta"), py::arg("bruteforce_selectivity_threshold"), py::arg("num_threads"))
        .def("batch_filter_search", &AcornIndexUint8::batch_filter_search);

        py::class_<AcornIndexFloat>(m, "AcornIndexFloat")
        .def(py::init<std::string, std::string, size_t, size_t, size_t, size_t, size_t, float, size_t>(),
            py::arg("filename"), py::arg("filter_filename"), py::arg("dataset_size"), py::arg("dim"), py::arg("M"), py::arg("gamma"), py::arg("m_beta"), py::arg("bruteforce_selectivity_threshold"), py::arg("num_threads"))
        .def("batch_filter_search", &AcornIndexFloat::batch_filter_search);
}
