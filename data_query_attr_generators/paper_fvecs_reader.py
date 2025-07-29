import numpy as np
import math
import pickle
import random

from scipy.sparse import lil_matrix, csc_matrix
from utils import fvecs_read

if __name__ == "__main__":
    query_matrix = fvecs_read("biganntest/data/paper/paper_query.fvecs", np.int32)
    dataset = fvecs_read("biganntest/data/paper/paper_base.fvecs", np.int32)

    # 20 data attrs, each drawn from zipf distribution
    random.seed(10)
    n_filters = 20
    data_filters = lil_matrix((dataset.shape[0], n_filters), dtype=np.int32)
    for j in range(n_filters):
        prob = math.pow(1 / (j + 2), 0.5)
        mask = np.random.rand(dataset.shape[0]) < prob
        data_filters[:, j] = mask.astype(np.int32).reshape(-1, 1)
    print("nonzero attributes:", data_filters.count_nonzero())
    data_filters_csc = csc_matrix(data_filters)
    pickle.dump(data_filters_csc, open("biganntest/data/paper/paper_data_attrs.pkl", "wb"))

    # Generate query templates following zipf distirbution
    total_queries = 10000
    min_filters = 2
    max_filters = 5
    zipf_coef = 2500

    sum_prob = sum(math.pow(1 / (j + 3), 0.65) for j in range(n_filters))
    probs = [math.pow(1 / (j + 3), 0.65) / sum_prob for j in range(n_filters)]
    filter_list = list(range(n_filters))

    random.seed(10)
    np.random.seed(10)
    q_list = []
    templates = {}
    curr_zipf_coef = zipf_coef

    while len(q_list) < total_queries:
        num_filters = random.randint(min_filters, max_filters)
        template = tuple(sorted(np.random.choice(filter_list, num_filters, replace=False, p=probs)))
        if template in templates:
            continue

        count = 0
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
        idx_set = set.intersection(*setlist)
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

    pickle.dump(csc_matrix(query_filters), open("biganntest/data/paper/paper_query_attrs.pkl", "wb"))
            