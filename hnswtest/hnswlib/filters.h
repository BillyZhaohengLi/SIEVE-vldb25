#pragma once

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
#include "roaring.hh"
#include "roaring.c"

#define XXH_INLINE_ALL
#include "xxhash.h"

namespace hnswlib {

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

struct QueryFilter {
    std::unordered_set<int32_t> _filters;
    std::pair<float, float> _first_pair;
    std::pair<float, float> _second_pair;
    bool _is_and = true;
    bool _is_range = false;

    QueryFilter(
        std::unordered_set<int32_t> filters,
        bool is_and
    ) : _filters(filters), _is_and(is_and), _is_range(false) {};

    QueryFilter(
        std::vector<float> filters,
        bool is_and
    ) : _first_pair(filters[0], filters[1]), _second_pair(filters[2], filters[3]), _is_and(is_and), _is_range(true) {};

    
    QueryFilter() {
        _first_pair = std::make_pair(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
        _second_pair = std::make_pair(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
    };

    bool is_single_filter() const {
        if (!_is_range) {
            // because the only possible negative value is -1, we can return the inverse of the sign bit of b
            return _filters.size() == 1;
        }
        return false;
    }

    bool operator==(const QueryFilter& rhs) const {
        if (!_is_range) {
            return _filters == rhs._filters && _is_and == rhs._is_and;
        }
        return _first_pair == rhs._first_pair && _second_pair == rhs._second_pair && _is_and == rhs._is_and;
    }

    bool is_subset(const QueryFilter& rhs) const {
        if (!_is_range) {
            // RHS is root partition
            if (rhs._filters.empty()) {
                return true;
            }

            if (_is_and != rhs._is_and) {
                return false;
            }

            if (_is_and) {
                // All elements of rhs must be in _filters
                return std::all_of(rhs._filters.begin(), rhs._filters.end(),
                                   [&](int32_t elem) { return _filters.count(elem); });
            } else {
                // All elements of _filters must be in rhs
                return std::all_of(_filters.begin(), _filters.end(),
                                   [&](int32_t elem) { return rhs._filters.count(elem); });
            }
        }
        return _first_pair.first >= rhs._first_pair.first && _first_pair.second <= rhs._first_pair.second &&
               _second_pair.first >= rhs._second_pair.first && _second_pair.second <= rhs._second_pair.second;
    }

    std::size_t hash() const {
        if (!_is_range) {
            int sum = 0;
            for (const auto& elem : _filters) {
                sum += elem;
            } 
            if (_is_and) {
                return sum;
            }
            return ~sum;
        }
        int sum = _first_pair.first + _first_pair.second + _second_pair.first + _second_pair.second;
        if (_is_and) {
            return sum;
        }
        return ~sum;
    }
};

struct QueryFilterHash
{
  std::size_t operator()(const QueryFilter& k) const
  {
    return k.hash();
  }
};

struct RoaringBitmapHash
{
  std::size_t operator()(const roaring::Roaring& k) const {
      return k.cardinality();
  }
};

inline size_t first_greater_than_or_equal_to(
    uint64_t n_points,
    float filter_value,
    const std::unique_ptr<float[]>& filter_values) {
    size_t left = 0, right = n_points;
    while (left < right) {
        size_t mid = left + (right - left) / 2;
        if (filter_values[mid] < filter_value) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    return left;
}

struct DatasetFilters{
    uint64_t n_points;
    uint64_t n_filters;
    uint64_t n_nonzero;
    size_t _num_threads;
    std::unique_ptr<uint64_t[]> row_offsets; // indices into data
    std::unique_ptr<uint32_t[]> row_indices; // the indices of the nonzero entries, which is actually all we need
    bool transposed = false;
    std::vector<roaring::Roaring*> filter_bvs;
    roaring::Roaring* base_bv;
    std::unordered_map<QueryFilter, roaring::Roaring*, QueryFilterHash> multifilters_bvs;

    // range predicate fields
    bool is_range = false;
    std::unique_ptr<float[]> first_row; 
    std::unique_ptr<float[]> second_row; 
    std::unordered_map<QueryFilter, roaring::Roaring*, QueryFilterHash> bvs;

    DatasetFilters() = default;

    /* mmaps filter data in csr form from filename */
    DatasetFilters(FILE* fp, size_t num_threads, bool is_range) : _num_threads(num_threads), is_range(is_range) {
        if (!is_range) {
            // opening file stream
            if (fp == NULL) {
                fprintf(stderr, "Error opening file");
                exit(1);
            }

            // reading in number of points, filters, and nonzeros
            fread(&n_points, sizeof(uint64_t), 1, fp);
            fread(&n_filters, sizeof(uint64_t), 1, fp);
            fread(&n_nonzero, sizeof(uint64_t), 1, fp);

            // reading in row offsets
            row_offsets = std::make_unique<uint64_t[]>(n_points + 1);
            fread(row_offsets.get(), sizeof(uint64_t), n_points + 1, fp);

            // reading in row indices
            row_indices = std::make_unique<uint32_t[]>(n_nonzero);
            fread(row_indices.get(), sizeof(int32_t), n_nonzero, fp);

            fclose(fp);

            for (uint64_t i = 0; i < n_points; i++) {
                std::sort(row_indices.get() + row_offsets[i], row_indices.get() + row_offsets[i + 1]);
            }
            
            // transpose_inplace();
            // std::cout << "transpose" << std::endl << std::flush;
            // make_bvs();
            // std::cout << "make bvs" << std::endl << std::flush;
        } else {
            // opening file stream
            if (fp == NULL) {
                fprintf(stderr, "Error opening file");
                exit(1);
            }

            std::cout << "start fread" << std::endl << std::flush;

            // reading in number of points, filters, and nonzeros
            fread(&n_filters, sizeof(uint64_t), 1, fp);
            fread(&n_points, sizeof(uint64_t), 1, fp);

            std::cout << "fread" << n_points << std::endl << std::flush;

            // reading in first row
            first_row = std::make_unique<float[]>(n_points);
            fread(first_row.get(), sizeof(float), n_points, fp);

            // reading in row indices
            second_row = std::make_unique<float[]>(n_points);
            fread(second_row.get(), sizeof(float), n_points, fp);

            fclose(fp);
        }
    }

    /* transposes the filters in place */
    void transpose_inplace() {
        if (!is_range) {
            std::cout << "Transposing (inplace)..." << std::endl;

            std::unique_ptr<uint64_t[]> new_row_offsets = std::make_unique<uint64_t[]>(n_filters + 1);
            std::unique_ptr<uint32_t[]> new_row_indices = std::make_unique<uint32_t[]>(n_nonzero);
    
            memset(new_row_offsets.get(), 0, (n_filters + 1) * sizeof(uint64_t)); // initializing to 0s
    
            std::cout << n_filters + 1 << std::endl;
            // counting points associated with each filter and scanning to get row offsets
            for (uint64_t i = 0; i < n_nonzero; i++) {
                new_row_offsets[row_indices[i] + 1]++;
            }
    
            // not a sequence so for now I'll just do it serially
            for (uint64_t i = 1; i < n_filters + 1; i++) {
                new_row_offsets[i] += new_row_offsets[i - 1];
            }
    
            // int64_t* tmp_offset = (int64_t*) malloc(n_filters * sizeof(int64_t)); // temporary array to keep track of where to put the next point in each filter
            std::unique_ptr<uint64_t[]> tmp_offset = std::make_unique<uint64_t[]>(n_filters);
            memset(tmp_offset.get(), 0, n_filters * sizeof(uint64_t)); // initializing to 0s
    
            // iterating over the data to fill in row indices
            for (uint64_t i = 0; i < n_points; i++) {
                int64_t start = row_offsets[i];
                int64_t end = row_offsets[i + 1];
                for (int64_t j = start; j < end; j++) {
                    int64_t f = row_indices[j];
                    int64_t index = new_row_offsets[f] + tmp_offset[f];
                    new_row_indices[index] = i;
                    tmp_offset[f]++;
                }
            }
    
            std::swap(this->n_points, this->n_filters);
    
            this->row_offsets = std::move(new_row_offsets);
            this->row_indices = std::move(new_row_indices);
    
            this->transposed = !transposed;
            return;
        }
    }

    void make_bvs() {
        if (!is_range) {
            filter_bvs.resize(n_points);
            ParallelFor(0, n_points, _num_threads, [&](size_t i, size_t) {
                auto start = row_offsets[i];
                auto end = row_offsets[i + 1];
                roaring::Roaring* query_bitset = new roaring::Roaring;
                if (end > start) {
                    query_bitset->addMany(end - start, row_indices.get() + start);
                }
                filter_bvs[i] = query_bitset;
            });
            base_bv = new roaring::Roaring;
            base_bv->addRange(0, n_filters);
        } else {
            base_bv = new roaring::Roaring;
            base_bv->addRange(0, n_points);
            bvs[QueryFilter()] = base_bv;
        }
    }

    void insert_multifilter_bv(QueryFilter q) {
        if (!is_range) {
            if (multifilters_bvs.find(q) != multifilters_bvs.end()) return;

            roaring::Roaring tmp_bv;
            bool first = true;
            for (const auto& elem : q._filters) {
                if (first) {
                    tmp_bv = *filter_bvs[elem];
                    first = false;
                } else {
                    if (q._is_and) {
                        tmp_bv &= *filter_bvs[elem];
                    } else {
                        tmp_bv |= *filter_bvs[elem];
                    }  
                }
            }
            multifilters_bvs[q] = new roaring::Roaring(tmp_bv);
        } else {
            auto first_start = first_greater_than_or_equal_to(n_points, q._first_pair.first, first_row);
            auto first_end = first_greater_than_or_equal_to(n_points, q._first_pair.second, first_row);
            auto second_start = first_greater_than_or_equal_to(n_points, q._second_pair.first, second_row);
            auto second_end = first_greater_than_or_equal_to(n_points, q._second_pair.second, second_row);

            roaring::Roaring tmp_bv;
            if (q._is_and) {
                tmp_bv.addRange(std::max(first_start, second_start), std::min(first_end, second_end));
            } else {
                tmp_bv.addRange(first_start, first_end);
                tmp_bv.addRange(second_start, second_end);
            }
            bvs[q] = new roaring::Roaring(tmp_bv);
        }
    }

    bool is_bitvector_cached(const QueryFilter& q) {
        if (!is_range) {
            // Single filter bitvectors are always available
            return q.is_single_filter() || multifilters_bvs.find(q) != multifilters_bvs.end();
        } else {
            return bvs.find(q) != bvs.end();
        }
    }

    roaring::Roaring* query_matches(QueryFilter q) {
        if (!is_range) {
            if (!transposed) {
                std::cout << "You are attempting to query a non-transposed csr_filter. This would require iterating over all the points in the dataset, which is almost certainly not what you want to do. Transpose this object." << std::endl;
                exit(1);
            }
            // Handle multi-filter queries
            if (q._filters.size() > 1) {
                if (!is_bitvector_cached(q)) {
                    insert_multifilter_bv(q);
                }
                return multifilters_bvs[q];
            }
            // Handle empty filter (root partition)
            if (q._filters.empty()) {
                return base_bv;
            }
            // Handle single filter
            return filter_bvs[*q._filters.begin()];
        } else {
            if (!is_bitvector_cached(q)) {
                insert_multifilter_bv(q);
            }
            return bvs[q];
        }
    }
};

std::vector<std::pair<QueryFilter, size_t>> tally_query_filters(
    const std::vector<QueryFilter>& query_filters,
    DatasetFilters* dataset_filters,
    size_t bitvector_cutoff
) {
    std::unordered_map<QueryFilter, size_t, QueryFilterHash> filter_counts;
    for (const auto& query_filter : query_filters) {
        // Skip filters with bitvector cardinality below cutoff
        if (dataset_filters->query_matches(query_filter)->cardinality() < bitvector_cutoff) {
            continue;
        }
        filter_counts[query_filter]++;
    }

    std::vector<std::pair<QueryFilter, size_t>> result(filter_counts.begin(), filter_counts.end());
    return result;
}

class BitMapFilter: public hnswlib::BaseFilterFunctor {
 public:
    BitMapFilter(roaring::Roaring* bitset): _bs(*bitset) {
    }
    bool operator()(hnswlib::labeltype label_id) {
        return _bs.contains(label_id);
    }
 private:
  roaring::Roaring& _bs;
};


class Predicate {
public:
  roaring::Roaring* _bitvector;
  QueryFilter _query_filter;
  size_t _cardinality;

  Predicate(DatasetFilters* dataset_filters, QueryFilter query_filter) : _bitvector(dataset_filters->query_matches(query_filter)), _query_filter(query_filter) {
      _cardinality = _bitvector->cardinality();
  }

  Predicate(size_t n_points) {
    _bitvector = new roaring::Roaring;
    _bitvector->addRange(0, n_points);
    _cardinality = _bitvector->cardinality();
  }

  Predicate() {}

  size_t cardinality() const {
      return _cardinality;
  }

  std::vector<uint32_t> matching_points() const {
    std::vector<uint32_t> tmp_vec;
    for (roaring::Roaring::const_iterator i = _bitvector->begin(); i != _bitvector->end(); i++) {
      tmp_vec.push_back(*i);
    }
    return tmp_vec;
  }

  bool is_bitvector_subset(const Predicate& rhs) const {
    return _bitvector->isSubset(*rhs._bitvector);
  }

  bool is_logical_subset(const Predicate& rhs) const {
    return _query_filter.is_subset(rhs._query_filter);
  }
};

} // namespace hnswlib
