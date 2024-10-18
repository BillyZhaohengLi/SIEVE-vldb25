#include "../../hnswlib/hnswlib.h"
#include "boost/dynamic_bitset.hpp"
#include <thread>
#include <bitset>
#include <chrono>
#include <unordered_map>
#include <set>
#include <queue>
#include <utility>

using namespace roaring;

// Multithreaded executor
// The helper function copied from python_bindings/bindings.cpp (and that itself is copied from nmslib)
// An alternative is using #pragme omp parallel for or any other C++ threading
template<class Function>
inline void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn) {
    if (numThreads <= 0) {
        numThreads = std::thread::hardware_concurrency();
    }

    if (numThreads == 1) {
        for (size_t id = start; id < end; id++) {
            fn(id, 0);
        }
    } else {
        std::vector<std::thread> threads;
        std::atomic<size_t> current(start);

        // keep track of exceptions in threads
        // https://stackoverflow.com/a/32428427/1713196
        std::exception_ptr lastException = nullptr;
        std::mutex lastExceptMutex;

        for (size_t threadId = 0; threadId < numThreads; ++threadId) {
            threads.push_back(std::thread([&, threadId] {
                while (true) {
                    size_t id = current.fetch_add(1);

                    if (id >= end) {
                        break;
                    }

                    try {
                        fn(id, threadId);
                    } catch (...) {
                        std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
                        lastException = std::current_exception();
                        /*
                         * This will work even when current is the largest value that
                         * size_t can fit, because fetch_add returns the previous value
                         * before the increment (what will result in overflow
                         * and produce 0 instead of current + 1).
                         */
                        current = end;
                        break;
                    }
                }
            }));
        }
        for (auto &thread : threads) {
            thread.join();
        }
        if (lastException) {
            std::rethrow_exception(lastException);
        }
    }
}


// Filter that allows labels divisible by divisor
class PickDivisibleIds: public hnswlib::BaseFilterFunctor {
unsigned int divisor = 1;
 public:
    PickDivisibleIds(unsigned int divisor): divisor(divisor) {
        assert(divisor != 0);
    }
    bool operator()(hnswlib::labeltype label_id) {
        return label_id % divisor == 0;
    }
};

// Filter that allows labels divisible by divisor
template <size_t bitsetsize>
class BitMapFilter: public hnswlib::BaseFilterFunctor {
unsigned int divisor = 1;
 public:
    BitMapFilter(std::bitset<bitsetsize>& bitset): _bs(bitset), total_evals(0) {
    }
    bool operator()(hnswlib::labeltype label_id) {
        return _bs[label_id] == 1;
    }
    int total_evals;
 private:
  std::bitset<bitsetsize> _bs;
};

float eval_recall(std::priority_queue<std::pair<float, hnswlib::labeltype>> result, std::set<int> gt) {
    float found = 0;
    for (; !result.empty(); result.pop())
    {
        hnswlib::labeltype label = result.top().second;
        // std::cout << "label: " << label << std::endl << std::flush;
        if (gt.find((int) label) != gt.end()) {
            found += 1;
        }
    }
    float rec = found / gt.size();
    return rec;
}

int main() {
    int dim = 50;               // Dimension of the elements
    const int max_elements = 10000;   // Maximum number of elements, should be known beforehand

    // historical queries
    Roaring* historical_workload = new Roaring[35];
    int large_step = 2000;
    for (int i = 0; i < max_elements; i += large_step) {
        Roaring query_bitset;
        for (int j = i; j < i + large_step; j++) {
            //query_bitset[j] = 1;
            query_bitset.add(j);
        }
        historical_workload[i / large_step] = query_bitset;        
    }

    int small_step = 1000;
    for (int i = 0; i < max_elements; i += small_step) {
        Roaring query_bitset;
        for (int j = i; j < i + small_step; j++) {
            //query_bitset[j] = 1;
            query_bitset.add(j);
        }
        historical_workload[max_elements / large_step + i / small_step] = query_bitset;       
    }

    // These should get pruned
    int tiny_step = 500;
    for (int i = 0; i < max_elements; i += tiny_step) {
        Roaring query_bitset;
        for (int j = i; j < i + tiny_step; j++) {
            //query_bitset[j] = 1;
            query_bitset.add(j);
        }
        historical_workload[max_elements / large_step + max_elements / small_step + i / tiny_step] = query_bitset;   
    }

    // Generate random data
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    float* data = new float[dim * max_elements];
    for (int i = 0; i < dim * max_elements; i++) {
        data[i] = distrib_real(rng);
    }

    std::cout << "finished generating data..." << std::endl << std::flush;

    // Initing index
    hnswlib::L2Space* space = new hnswlib::L2Space(dim);
    int M = 32;                 // Tightly connected with internal dimensionality of the data
                        // strongly affects the memory consumption
    int ef_construction = 40;  // Controls index search speed/build speed tradeoff
    int num_threads = 20;       // Number of threads for operations with index
    int k = 10;

    hnswlib::PartitionedHNSW* partitioned_hnsw = new hnswlib::PartitionedHNSW(data, space, historical_workload, 35, max_elements, max_elements, dim);

    // Serve some queries
    const int num_queries = 100;

    float* query = new float[num_queries * max_elements];
    for (int i = 0; i < num_queries * max_elements; i++) {
        query[i] = distrib_real(rng);
    }

    // // test bruteforce
    // for (int j = 0; j < num_queries; j++) {
    //     std::cout << "bruteforcing outside: " << query + dim * j << std::endl << std::flush;
    //     std::priority_queue<std::pair<float, int>> max_priority_queue;
    //     for (int i = 0; i < max_elements; i++) {
    //         if (!historical_workload[16][i]) {
    //             continue;
    //         }
    //         // if (!query_filter.contains(i)) {
    //         //     continue;
    //         // }
    //         std::cout << "Current location: " << (data + dim * i) << " dim: " << dim << " i: " << i << std::flush;
    //         float dist = alg_hnsw->fstdistfunc_(query + dim * j, data + dim * i, alg_hnsw->dist_func_param_);
    //         if (max_priority_queue.size() < k) {
    //             max_priority_queue.push(std::make_pair(dist, i));
    //         } else if (dist < max_priority_queue.top().first) {
    //             max_priority_queue.pop();
    //             max_priority_queue.push(std::make_pair(dist, i));
    //         }
    //     }
    // }
            

    for (int i = 0; i < num_queries; i++) {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = partitioned_hnsw->searchKnn(query + i * dim, k, historical_workload[0]);
    }

    // // Compute ground truth
    // std::unordered_map<int, std::set<int>> ground_truths;
    // auto start = std::chrono::high_resolution_clock::now();
    // for (int row = 0; row < num_queries; row++) {
    //     std::priority_queue<std::pair<float, int>> max_priority_queue;
    //     for (int i = 0; i < max_elements; i++) {
    //         float dist = partitioned_hnsw->_root->_hnsw->fstdistfunc_((data + dim * row), (data + dim * i), partitioned_hnsw->_root->_hnsw->dist_func_param_);
    //         if (max_priority_queue.size() < k) {
    //             max_priority_queue.push(std::make_pair(dist, i));
    //         } else if (dist < max_priority_queue.top().first) {
    //             max_priority_queue.pop();
    //             max_priority_queue.push(std::make_pair(dist, i));
    //         }
    //     }
    //     std::set<int> top_indices;
    //     for (; !max_priority_queue.empty(); max_priority_queue.pop()) {
    //         top_indices.insert(max_priority_queue.top().second);
    //     }
    //     ground_truths.insert(std::make_pair(row, top_indices));
    // }
    // auto end = std::chrono::high_resolution_clock::now();
    // std::cout << "prefilter search on entire dataset QPS: " << num_queries / std::chrono::duration<double>(end - start).count() << std::endl << std::flush;

    delete[] data;
    delete partitioned_hnsw;
    return 0;
}
