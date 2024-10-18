#include "../../hnswlib/hnswlib.h"
#include <thread>
#include <bitset>

// Filter that allows labels divisible by divisor
template <size_t bitsetsize>
class BitMapFilter: public hnswlib::BaseFilterFunctor {
unsigned int divisor = 1;
 public:
    BitMapFilter(std::bitset<bitsetsize>& bitset): _bs(bitset) {
    }
    bool operator()(hnswlib::labeltype label_id) {
        return _bs[label_id] == 1;
    }
 private:
  std::bitset<bitsetsize> _bs;
};

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


int main() {
    int dim = 16;               // Dimension of the elements
    const int max_elements = 1000000;   // Maximum number of elements, should be known beforehand
    int M = 16;                 // Tightly connected with internal dimensionality of the data
                                // strongly affects the memory consumption
    int ef_construction = 200;  // Controls index search speed/build speed tradeoff
    int num_threads = 8;       // Number of threads for operations with index

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

    // Add data to index
    ParallelFor(0, max_elements, num_threads, [&](size_t row, size_t threadId) {
        alg_hnsw->addPoint((void*)(data + dim * row), row);
    });

    // Create bitset filters
    std::bitset<max_elements> bitset_entire_graph;
    ParallelFor(0, max_elements, num_threads, [&](size_t row, size_t threadId) {
        bitset_entire_graph[row] = 1;
    });
    BitMapFilter<max_elements> BitMapFilterEntireGraph(bitset_entire_graph);

    auto start = std::chrono::high_resolution_clock::now();
    // Query the elements for themselves and measure recall
    int k = 10;
    std::vector<hnswlib::labeltype> neighbors(max_elements * k);
    ParallelFor(0, max_elements, num_threads, [&](size_t row, size_t threadId) {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data + dim * row, k);
        for (int i = 0; i < k; i++) {
            hnswlib::labeltype label = result.top().second;
            result.pop();
            neighbors[row * k + i] = label;
        }
    });
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration<double>(end - start).count() << "s" << std::endl;
    delete[] data;
    delete alg_hnsw;
    return 0;
}
