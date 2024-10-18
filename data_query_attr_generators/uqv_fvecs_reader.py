import random
import math
import numpy as np
from scipy.sparse import csc_matrix, lil_matrix, csr_matrix
import pickle
from heapq import heapify, heappush, heappop 

def write_sparse_matrix(mat, fname):
    """ write a CSR matrix in the spmat format """
    with open(fname, "wb") as f:
        sizes = np.array([mat.shape[0], mat.shape[1], mat.nnz], dtype='int64')
        sizes.tofile(f)
        indptr = mat.indptr.astype('int64')
        indptr.tofile(f)
        mat.indices.astype('int32').tofile(f)
        mat.data.astype('float32').tofile(f)

def read_sparse_matrix_fields(fname):
    """ read the fields of a CSR matrix without instanciating it """
    with open(fname, "rb") as f:
        sizes = np.fromfile(f, dtype='int64', count=3)
        nrow, ncol, nnz = sizes
        indptr = np.fromfile(f, dtype='int64', count=nrow + 1)
        assert nnz == indptr[-1]
        indices = np.fromfile(f, dtype='int32', count=nnz)
        assert np.all(indices >= 0) and np.all(indices < ncol)
        data = np.fromfile(f, dtype='float32', count=nnz)
        return data, indices, indptr, ncol

def mmap_sparse_matrix_fields(fname):
    """ mmap the fields of a CSR matrix without instanciating it """
    with open(fname, "rb") as f:
        sizes = np.fromfile(f, dtype='int64', count=3)
        nrow, ncol, nnz = sizes
    ofs = sizes.nbytes
    indptr = np.memmap(fname, dtype='int64', mode='r', offset=ofs, shape=nrow + 1)
    ofs += indptr.nbytes
    indices = np.memmap(fname, dtype='int32', mode='r', offset=ofs, shape=nnz)
    ofs += indices.nbytes
    data = np.memmap(fname, dtype='float32', mode='r', offset=ofs, shape=nnz)
    return data, indices, indptr, ncol

def read_sparse_matrix(fname, do_mmap=False):
    """ read a CSR matrix in spmat format, optionally mmapping it instead """
    if not do_mmap:
        data, indices, indptr, ncol = read_sparse_matrix_fields(fname)
    else:
        data, indices, indptr, ncol = mmap_sparse_matrix_fields(fname)

    return csr_matrix((data, indices, indptr), shape=(len(indptr) - 1, ncol))

def fvecs_read(filename, dtype, c_contiguous=True):
    fv = np.fromfile(filename, dtype=dtype)

    if fv.size == 0:
       return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    # print("dim:", fv.view(np.int32))
    # print(len(fv.view(np.int32)))
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv.view(np.int32)[:, 0] == dim):
       raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv[:, 1:]
    if c_contiguous:
       fv = fv.copy()
    print(fv.shape)
    return fv


def read_fbin(filename, start_idx=0, chunk_size=None):
    """ Read *.fbin file that contains float32 vectors
    Args:
        :param filename (str): path to *.fbin file
        :param start_idx (int): start reading vectors from this index
        :param chunk_size (int): number of vectors to read.
                                 If None, read all vectors
    Returns:
        Array of float32 vectors (numpy.ndarray)
    """
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        nvecs = (nvecs - start_idx) if chunk_size is None else chunk_size
        arr = np.fromfile(f, count=nvecs * dim, dtype=np.float32,
                          offset=start_idx * 4 * dim)
    return arr.reshape(nvecs, dim)

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


if __name__ == "__main__":
    query = fvecs_read("~/bigann/biganntest/data/uqv/uqv_query.fvecs", np.int32)
    dataset = fvecs_read("~/bigann/biganntest/data/uqv/uqv_base.fvecs", np.int32)

    n_filters = 200000
    random.seed(10)
    data_filters_lil_matrix = lil_matrix((dataset.shape[0], n_filters), dtype=np.int32)
    for j in range(n_filters):
        if j % 1000 == 0:
            print(j)
        target = 1 / (j + 3)
        for i in range(num_points):
            if random.random() < target:
                data_filters_lil_matrix[i, j] = 1
    print("nonzero:", data_filters_lil_matrix.count_nonzero())
    data_filters_csc_matrix = csc_matrix(data_filters_lil_matrix)
    pickle.dump(data_filters_csc_matrix, open("~/bigann/biganntest/data/uqv/uqv_query_attrs.pkl", "wb"))
    spmat = pickle.load(open("~/bigann/biganntest/data/uqv/uqv_query_attrs.pkl", "rb"))
    filter_list = [i for i in range(200000)]

    sum_prob = sum([math.pow(1/(j+3), 0.75) for j in range(200000)])
    probs = [math.pow(1/(j+3), 0.75) / sum_prob for j in range(200000)]

    count_list = [i for i in range(3, max_filters)]
    count_sum_prob = sum([1/(j+3) for j in range(3, max_filters)])
    count_probs = [1/(j+3) / count_sum_prob for j in range(3, max_filters)]

    # zipf distribution for queries
    random.seed(10)
    np.random.seed(10)
    sum = 10000
    q_list = []
    templates = {}
    zipf_coef = 2500
    while len(q_list) < sum:
        i = 0
        # Choose a query template
        num_filters = np.random.choice(count_list, size=1, replace=False,p=count_probs)
        template = np.random.choice(filter_list, size=num_filters[0], replace=False, p=probs)
        if frozenset(template) in templates:
            continue
        # template = random.sample(filter_list,20,weights=[1/(j+3) for j in range(200000)])
        while i < sum / zipf_coef / 9:
            i += 1
            q_list.append(template)
            if len(q_list) >= sum:
                break
        zipf_coef -= 1
        templates[frozenset(template)] = i
    print(zipf_coef)

    spmat = csc_matrix(spmat)
    sel = 0
    for k, v in templates.items():
        print(k)
        idx_set = set()
        for f in k:
            idx_set = idx_set.union(set(spmat.indices[spmat.indptr[f]:spmat.indptr[f+1]]))
        sel += len(idx_set) * v
        print(len(idx_set))
    print("selectivity:", sel)

    random.shuffle(q_list)

    query_filters_lil_matrix = lil_matrix((10000, 200000), dtype=np.int32)
    for i in range(10000):
        for j in q_list[i]:
            query_filters_lil_matrix[i,j] = 1
    query_filters_csc_matrix = csc_matrix(query_filters_lil_matrix)
    
    pickle.dump(query_filters_csc_matrix, open("~/bigann/biganntest/data/uqv/uqv_query_attrs.pkl", "wb"))
