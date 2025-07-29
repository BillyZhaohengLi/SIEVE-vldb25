import bisect
import numpy as np
import pickle
import random

from utils import fvecs_read

if __name__ == "__main__":
    query_matrix = fvecs_read("biganntest/data/sift2/sift_query.fvecs", np.int32)
    dataset = fvecs_read("biganntest/data/sift2/sift_base.fvecs", np.int32)

    # 2 attrs, each drawn from normal distribution
    # Generate and sort filter attributes for each filter
    np.random.seed(10)
    n_filters = 2
    data_filters_matrix = np.sort(
        np.random.normal(loc=0, scale=1, size=(n_filters, dataset.shape[0])), axis=1).astype(np.float32)
    pickle.dump(np.transpose(data_filters_matrix), open("biganntest/data/sift2/sift_data_attrs.pkl", "wb"))

# Generate query templates using a Zipf-like distribution
    random.seed(10)  # Alternate dataset uses seed=42
    np.random.seed(10)  # Alternate dataset uses seed=42
    total_queries = 10000
    templates = {}
    zipf_coef = 200
    queries_generated = 0

    # Generate query templates
    while queries_generated < total_queries:
        # Generate a query template
        template = sorted(np.random.normal(loc=0, scale=1, size=4))
        
        # Randomly permute the sorted values
        idx = 3 if random.random() > 0.5 else 2
        value = template.pop(idx)
        template.insert(1, value)
        
        if random.random() > 0.5:
            template = template[2:] + template[:2]
        
        template = tuple(template)

        if template in templates:
            continue

        count = 0
        max_count = total_queries / max(zipf_coef, 1) / 5
        while count < max_count and queries_generated < total_queries:
            count += 1
            queries_generated += 1
        templates[template] = count
        zipf_coef -= 1

    print("unique templates:", len(templates))

    # Calculate selectivity based on the generated templates
    sels = []
    total_selectivity = 0

    for template, count in templates.items():
        f0_left = bisect.bisect_left(data_filters_matrix[0], template[0])
        f0_right = bisect.bisect_right(data_filters_matrix[0], template[1])
        f1_left = bisect.bisect_left(data_filters_matrix[1], template[2])
        f1_right = bisect.bisect_right(data_filters_matrix[1], template[3])

        selectivity = min(f0_right, f1_right) - max(f0_left, f1_left)

        total_selectivity += count * selectivity
        sels.extend([selectivity] * count)

    print("Avg. selectivity:", total_selectivity / total_queries)

    sels.sort()
    print("min:", sels[0])
    print("p20:", sels[2000])
    print("p40:", sels[4000])
    print("p60:", sels[6000])
    print("p80:", sels[8000])
    print("max:", sels[9999])

    # Create query filter matrix
    tmp_matrix = [list(template) for template, count in templates.items() for _ in range(count)]

    # Shuffle query filter matrix
    random.shuffle(tmp_matrix)
    random.shuffle(tmp_matrix)
    pickle.dump(np.array(tmp_matrix, dtype = np.float32), open("biganntest/data/sift2/sift_query_attrs.pkl", "wb"))