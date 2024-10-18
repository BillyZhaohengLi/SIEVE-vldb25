#include "../../hnswlib/hnswlib.h"
#include <thread>
#include <bitset>
#include <chrono>
#include <unordered_map>
#include <set>
#include <queue>
#include <utility>
#include <iostream>
#include <dirent.h>

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


int main() {
    std::vector<float> bf_time;
    std::vector<float> hnsw_time;
    for (int max_elements = 50; max_elements <= 1000; max_elements += 50) {
        int dim = 192;               // Dimension of the elements
        // int max_elements = 500;   // Maximum number of elements, should be known beforehand
        int num_queries = 1000;   // Maximum number of elements, should be known beforehand
        int M = 64;                 // Tightly connected with internal dimensionality of the data
                            // strongly affects the memory consumption
        int ef_construction = 200;  // Controls index search speed/build speed tradeoff
        int k = 10;
    
        // Initing index
        hnswlib::L2Space space(dim);
        hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
    
        // Generate random data
        std::mt19937 rng;
        rng.seed(47);
        std::uniform_real_distribution<> distrib_real;
        float* data = new float[dim * max_elements];
        for (int i = 0; i < dim * max_elements; i++) {
            data[i] = distrib_real(rng);
        }
    
        // Generate random queries
        float* query = new float[dim * num_queries];
        for (int i = 0; i < dim * num_queries; i++) {
            query[i] = distrib_real(rng);
        }
    
        // Add to index
        for (int row = 0; row < max_elements; row++) {
            alg_hnsw->addPoint((void*)(data + dim * row), row);
        }
    
        
    
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_queries; i++) {
            alg_hnsw->searchKnn(query + dim * i, k);
        }
        auto end = std::chrono::high_resolution_clock::now();
        hnsw_time.push_back(std::chrono::duration<double>(end - start).count());
        std::cout << "HNSW time: " << std::chrono::duration<double>(end - start).count() << std::endl << std::flush;
    
        // Bruteforce search
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_queries; i++) {
            std::priority_queue<std::pair<float, size_t>> max_priority_queue;
            for (int j = 0; j < max_elements; j++) {
                int dist = alg_hnsw->fstdistfunc_((query + dim * i), (data + dim * j), alg_hnsw->dist_func_param_);
                if (max_priority_queue.size() < k) {
                    max_priority_queue.push(std::make_pair(dist, j));
                } else if (dist < max_priority_queue.top().first) {
                    max_priority_queue.pop();
                    max_priority_queue.push(std::make_pair(dist, j));
                }
            }
        }
        end = std::chrono::high_resolution_clock::now();
        bf_time.push_back(std::chrono::duration<double>(end - start).count());
        std::cout<< "Bruteforce time: " << std::chrono::duration<double>(end - start).count() << std::endl << std::flush;
    
        delete[] data;
        delete alg_hnsw;
    }
    for (int i = 0; i < hnsw_time.size(); i++) {
        std::cout << hnsw_time[i] << ",";
    }
    std::cout << std::endl << std::flush;
    for (int i = 0; i < bf_time.size(); i++) {
        std::cout << bf_time[i] << ",";
    }
    std::cout << std::endl << std::flush;
    return 0;
}
