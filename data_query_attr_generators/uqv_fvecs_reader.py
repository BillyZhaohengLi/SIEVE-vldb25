import numpy as np
import math
import pickle
import random

from scipy.sparse import lil_matrix, csc_matrix
from utils import fvecs_read

if __name__ == "__main__":
    query = fvecs_read("biganntest/data/uqv/uqv_query.fvecs", np.int32)
    dataset = fvecs_read("biganntest/data/uqv/uqv_base.fvecs", np.int32)

    # 200K data attrs, each drawn from zipf distribution
    random.seed(10)
    n_filters = 20
    data_filters = lil_matrix((dataset.shape[0], n_filters), dtype=np.int32)
    for j in range(n_filters):
        prob = 1 / (j + 3)
        mask = np.random.rand(dataset.shape[0]) < prob
        data_filters[:, j] = mask.astype(np.int32).reshape(-1, 1)
    print("Nonzero entries:", data_filters_lil_matrix.count_nonzero())
    data_filters_csc = csc_matrix(data_filters_lil_matrix)
    pickle.dump(data_filters_csc, open("biganntest/data/uqv/uqv_query_attrs.pkl", "wb"))

    filter_list = list(range(200000))

    # Precompute zipf probabilities for filter selection 
    probs = np.array([math.pow(1/(j+3), 0.75) for j in filter_list])
    probs /= probs.sum()

    # Probabilities for number of filters per query
    max_filters = 10
    count_list = list(range(3, max_filters))
    count_probs = np.array([1/(j+3) for j in count_list])
    count_probs /= count_probs.sum()

    # Generate query templates using Zipf distribution
    random.seed(10)  # Alternate dataset uses seed=42
    np.random.seed(10)  # Alternate dataset uses seed=42
    total_queries = 10000
    q_list = []
    templates = {}
    zipf_coef = 2500

    while len(q_list) < total_queries:
        num_filters = random.randint(min_filters, max_filters)
        template = tuple(sorted(np.random.choice(filter_list, num_filters, replace=False, p=probs)))
        if template in templates:
            continue

        ount = 0
        max_count = total_queries / curr_zipf_coef / 9
        while count < max_count and len(q_list) < total_queries:
            count += 1
            q_list.append(template)
        curr_zipf_coef = max(1, curr_zipf_coef - 1)
        templates[template] = count

    print("unique templates:", len(templates))

    sels = []
    total_selectivity = 0
    for template, count in templates.items():
        setlist = [set(data_filters_csc.indices[data_filters_csc.indptr[f]:data_filters_csc.indptr[f+1]]) for f in template]
        selectivity = len(idx_set)
        total_selectivity += len(idx_set) * count
        sels.extend([selectivity] * count)

    print("Avg. selectivity:", total_selectivity / total_queries)

    sels.sort()
    print("min:", sels[0])
    print("p20:", sels[2000])
    print("p40:", sels[4000])
    print("p60:", sels[6000])
    print("p80:", sels[8000])
    print("max:", sels[9999])

    # Convert query list to sparse matrix
    random.shuffle(q_list)
    query_filters = lil_matrix((total_queries, n_filters), dtype=np.int32)
    for i, template in enumerate(q_list):
        query_filters.rows[i] = list(template)
        query_filters.data[i] = [1] * len(template)
    
    pickle.dump(csc_matrix(query_filters), open("biganntest/data/uqv/uqv_query_attrs.pkl", "wb"))
