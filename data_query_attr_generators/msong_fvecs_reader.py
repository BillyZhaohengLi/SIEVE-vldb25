import numpy as np
import math
import pickle
import random

from scipy.sparse import lil_matrix, csc_matrix
from utils import fvecs_read

if __name__ == "__main__":
    query_matrix = fvecs_read("biganntest/data/msong/msong_query.fvecs", np.int32)
    dataset = fvecs_read("biganntest/data/msong/msong_base.fvecs", np.int32)

    # 19 attrs, each with (i * 0.05) selectivity, 1 <= i <= 19
    random.seed(10)
    n_filters = 19
    data_filters = lil_matrix((dataset.shape[0], 19), dtype=np.int32)
    for i in range(n_filters):
        selectivity = (i + 1) * 0.05
        mask = np.random.rand(dataset.shape[0]) <= selectivity
        data_filters[:, i] = mask.astype(np.int32)
    pickle.dump(data_filters, open("biganntest/data/msong/msong_data_attrs.pkl", "wb"))

    # 8 queries for each attr; other 200 - 19 * 8 = 48 queries have no filters.
    query_filters = lil_matrix((query_matrix.shape[0], n_filters))
    for i in range(query_matrix.shape[0]):
        if i % 25 < n_filters:
            query_filters[i, i % 25] = 1

    pickle.dump(csc_matrix(query_filters), open("biganntest/data/msong/msong_query_attrs.pkl", "wb"))

