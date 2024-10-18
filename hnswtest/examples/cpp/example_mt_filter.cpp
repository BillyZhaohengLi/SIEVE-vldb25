#include "../../hnswlib/hnswlib.h"
#include <thread>
#include <bitset>
#include <chrono>
#include <unordered_map>
#include <set>
#include <queue>
#include <utility>

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
    int dim = 192;               // Dimension of the elements
    const int max_elements = 200000;   // Maximum number of elements, should be known beforehand
    const int cluster_size = 200000;

    // Initing index
    hnswlib::L2Space space(dim);
    int M = 64;                 // Tightly connected with internal dimensionality of the data
                        // strongly affects the memory consumption
    int ef_construction = 200;  // Controls index search speed/build speed tradeoff
    int num_threads = 20;       // Number of threads for operations with index
    int k = 100;

    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);

    // Generate random data
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    float* data = new float[dim * max_elements];
    for (int i = 0; i < dim * max_elements; i++) {
        data[i] = distrib_real(rng);
    }

    // // Cluster some points
    // for (int i = 0; i < cluster_size; i++) {
    //     for (int j = 0; j < dim; j++) {
    //         data[dim * i + j] = distrib_real(rng) * 0.4 + data[dim * cluster_size + j];
    //     }
    // }
    

    std::cout << "Finish generating data" << std::endl << std::flush;

    // Add data to index
    ParallelFor(0, max_elements, num_threads, [&](size_t row, size_t threadId) {
        alg_hnsw->addPoint((void*)(data + dim * row), row);
    });

    std::cout << "Finish constructing overall graph" << std::endl << std::flush;
    
    const int num_queries = 100;

    for (float a_selectivity = 1; a_selectivity <= 1.01; a_selectivity += 0.1) {
        std::cout << "A selectivity: " << a_selectivity << "-----------------------" << std::endl << std::flush;
        int a_elements = (int) (a_selectivity * max_elements);

        // Compute ground truth
        std::unordered_map<int, std::set<int>> ground_truths;
        auto start = std::chrono::high_resolution_clock::now();
        ParallelFor(0, num_queries, num_threads, [&](size_t row, size_t threadId) {
            std::priority_queue<std::pair<float, int>> max_priority_queue;
            for (int i = 0; i < a_elements; i++) {
                float dist = alg_hnsw->fstdistfunc_((data + dim * row), (data + dim * i), alg_hnsw->dist_func_param_);
                if (max_priority_queue.size() < k) {
                    max_priority_queue.push(std::make_pair(dist, i));
                } else if (dist < max_priority_queue.top().first) {
                    max_priority_queue.pop();
                    max_priority_queue.push(std::make_pair(dist, i));
                }
            }
            std::set<int> top_indices;
            for (; !max_priority_queue.empty(); max_priority_queue.pop()) {
                top_indices.insert(max_priority_queue.top().second);
            }
            ground_truths.insert(std::make_pair(row, top_indices));
        });
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "prefilter search on entire dataset QPS: " << num_queries / std::chrono::duration<double>(end - start).count() << std::endl << std::flush;

        // unfiltered search on A graph
        // Create bitset filters
        std::bitset<max_elements> bitset_entire_graph;
        for (int i = 0; i < a_elements; i++) {
            bitset_entire_graph[i] = 1;
        }
        BitMapFilter<max_elements> BitMapFilterEntireGraph(bitset_entire_graph);

        // Construct A only index
        // hnswlib::HierarchicalNSW<float>* alg_hnsw_a = new hnswlib::HierarchicalNSW<float>(&space, a_elements, M, ef_construction);
        // ParallelFor(0, a_elements, num_threads, [&](size_t row, size_t threadId) {
        //     alg_hnsw_a->addPoint((void*)(data + dim * row), row);
        // });

        // ACORN-1 search on entire graph
        float avg_recall_acorn1 = 0;
        start = std::chrono::high_resolution_clock::now();
        ParallelFor(0, num_queries, num_threads, [&](size_t row, size_t threadId) {
            size_t row2 = max_elements - row;
            std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnnAcorn(data + dim * row, k, &BitMapFilterEntireGraph);
            avg_recall_acorn1 += eval_recall(result, ground_truths[row]);
        });
        end = std::chrono::high_resolution_clock::now();
        std::cout << "ACORN-1 search on entire dataset recall: " << avg_recall_acorn1 / num_queries << std::endl << std::flush;
        std::cout << "ACORN-1 search on entire dataset QPS: " << num_queries / std::chrono::duration<double>(end - start).count() << std::endl << std::flush;

        // HNSW search on entire graph
        float avg_recall_hnsw = 0;
        start = std::chrono::high_resolution_clock::now();
        ParallelFor(0, num_queries, num_threads, [&](size_t row, size_t threadId) {
            size_t row2 = max_elements - row;
            std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data + dim * row, k, &BitMapFilterEntireGraph);
            avg_recall_hnsw += eval_recall(result, ground_truths[row]);
        });
        end = std::chrono::high_resolution_clock::now();
        std::cout << "HNSW search on entire dataset recall: " << avg_recall_hnsw / num_queries << std::endl << std::flush;
        std::cout << "HNSW search on entire dataset QPS: " << num_queries / std::chrono::duration<double>(end - start).count() << std::endl << std::flush;

        // HNSW in place filter search on entire graph
        float avg_recall_in_place_filter = 0;
        start = std::chrono::high_resolution_clock::now();
        ParallelFor(0, num_queries, num_threads, [&](size_t row, size_t threadId) {
            size_t row2 = max_elements - row;
            std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnnInPlaceFilter(data + dim * row, k, &BitMapFilterEntireGraph);
            avg_recall_in_place_filter += eval_recall(result, ground_truths[row]);
        });
        end = std::chrono::high_resolution_clock::now();
        std::cout << "HNSW in place filter search on entire dataset recall: " << avg_recall_in_place_filter / num_queries << std::endl << std::flush;
        std::cout << "HNSW in place filter search on entire dataset QPS: " << num_queries / std::chrono::duration<double>(end - start).count() << std::endl << std::flush;

        // start = std::chrono::high_resolution_clock::now();
        // ParallelFor(0, num_queries, num_threads, [&](size_t row, size_t threadId) {
        //     std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw_a->searchKnn(data + dim * row, k, &BitMapFilterEntireGraph);
        // });
        // end = std::chrono::high_resolution_clock::now();
        // std::cout << "filtered search on A graph time: " << std::chrono::duration<double>(end - start).count() << std::endl << std::flush;
        // std::cout << "filter evals: " << BitMapFilterEntireGraph.total_evals << std::endl << std::flush;

        // Unfiltered search on A only graph
        // float avg_recall_unfiltered_a = 0;
        // start = std::chrono::high_resolution_clock::now();
        // ParallelFor(0, num_queries, num_threads, [&](size_t row, size_t threadId) {
        //     size_t row2 = max_elements - row;
        //     std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw_a->searchKnn(data + dim * row, k);
        //     avg_recall_unfiltered_a += eval_recall(result, ground_truths[row]);
        // });
        // end = std::chrono::high_resolution_clock::now();
        // std::cout << "unfiltered search on A graph recall: " << avg_recall_unfiltered_a / num_queries << std::endl << std::flush;
        // std::cout << "unfiltered search on A graph QPS: " << num_queries / std::chrono::duration<double>(end - start).count() << std::endl << std::flush;

        // float avg_recall_dummy_filter = 0;
        // start = std::chrono::high_resolution_clock::now();
        // ParallelFor(0, num_queries, num_threads, [&](size_t row, size_t threadId) {
        //     size_t row2 = max_elements - row;
        //     std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw_a->searchKnn(data + dim * row, k, &BitMapFilterEntireGraph);
        //     avg_recall_dummy_filter += eval_recall(result, ground_truths[row]);
        // });
        // end = std::chrono::high_resolution_clock::now();
        // std::cout << "dummy filter search on A graph recall: " << avg_recall_dummy_filter / num_queries << std::endl << std::flush;
        // std::cout << "dummy filter search on A graph QPS: " << num_queries / std::chrono::duration<double>(end - start).count() << std::endl << std::flush;
        // delete alg_hnsw_a;

        // for (float a_or_b_selectivity = 1; a_or_b_selectivity >= 0.09 & a_selectivity / a_or_b_selectivity < 1; a_or_b_selectivity -= 0.1) {
        //     int a_or_b_elements = (int) (a_selectivity / a_or_b_selectivity * max_elements);
        //     std::cout << " A in A OR B selectivity: " << a_or_b_selectivity << "-----------------------" << std::endl << std::flush;

        //     // Construct A OR B index
        //     hnswlib::HierarchicalNSW<float>* alg_hnsw_ab = new hnswlib::HierarchicalNSW<float>(&space, a_or_b_elements, M, ef_construction);
        //     ParallelFor(0, a_or_b_elements, num_threads, [&](size_t row, size_t threadId) {
        //         alg_hnsw_ab->addPoint((void*)(data + dim * row), row);
        //     });
        
        //     // ACORN-1 search on A or B graph
        //     float avg_recall_acorn_a_or_b = 0;
        //     auto start = std::chrono::high_resolution_clock::now();
        //     ParallelFor(0, num_queries, num_threads, [&](size_t row, size_t threadId) {
        //         size_t row2 = max_elements - row; 
        //         std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw_ab->searchKnnAcorn(data + dim * row, k, &BitMapFilterEntireGraph);
        //         avg_recall_acorn_a_or_b += eval_recall(result, ground_truths[row]);
        //     });
        //     auto end = std::chrono::high_resolution_clock::now();

        //     std::cout << "ACORN-1 search on A or B graph recall: " << avg_recall_acorn_a_or_b / num_queries << std::endl << std::flush;
        //     std::cout << "ACORN-1 search on A or B graph QPS: " << num_queries / std::chrono::duration<double>(end - start).count() << std::endl << std::flush;
        
        //     // HNSW search on A or B graph
        //     float avg_recall_hnsw_a_or_b = 0;
        //     start = std::chrono::high_resolution_clock::now();
        //     ParallelFor(0, num_queries, num_threads, [&](size_t row, size_t threadId) {
        //         size_t row2 = max_elements - row; 
        //         std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw_ab->searchKnn(data + dim * row, k, &BitMapFilterEntireGraph);
        //         avg_recall_hnsw_a_or_b += eval_recall(result, ground_truths[row]);
        //     });
        //     end = std::chrono::high_resolution_clock::now();
        //     std::cout << "HNSW postfilter search on A or B graph recall: " << avg_recall_hnsw_a_or_b / num_queries << std::endl << std::flush;
        //     std::cout << "HNSW postfilter search on A or B graph QPS: " << num_queries / std::chrono::duration<double>(end - start).count() << std::endl << std::flush;

        //     // HNSW search on A or B graph
        //     float avg_recall_hnsw_a_or_b_inplace = 0;
        //     start = std::chrono::high_resolution_clock::now();
        //     ParallelFor(0, num_queries, num_threads, [&](size_t row, size_t threadId) {
        //         size_t row2 = max_elements - row; 
        //         std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw_ab->searchKnnInPlaceFilter(data + dim * row, k, &BitMapFilterEntireGraph);
        //         avg_recall_hnsw_a_or_b_inplace += eval_recall(result, ground_truths[row]);
        //     });
        //     end = std::chrono::high_resolution_clock::now();
        //     std::cout << "HNSW inplace search on A or B graph recall: " << avg_recall_hnsw_a_or_b_inplace / num_queries << std::endl << std::flush;
        //     std::cout << "HNSW inplace search on A or B graph QPS: " << num_queries / std::chrono::duration<double>(end - start).count() << std::endl << std::flush;
                    
        //     delete alg_hnsw_ab;
        // }
    }

    delete[] data;
    delete alg_hnsw;
    return 0;
}
