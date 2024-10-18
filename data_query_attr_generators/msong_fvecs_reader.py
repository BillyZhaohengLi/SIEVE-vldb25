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

def setup_dataset(dataset, queries):
    # Generate data filters
    num_points = dataset.shape[0]
    dim = dataset.shape[1]

    # One-hot attributes
    # Following yfcc-10M, 200000 keywords as tags following harmonic distribution
    # i.e., the nth tag has probability 1/n of being assigned to a vector
    # We start at n=3.

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
    pickle.dump(data_filters_csc_matrix, open("/home/zl20/bigann/biganntest/data/msong/msong_query_attrs.pkl", "wb"))
    

    # n_filters = 200000
    # random.seed(10)
    # data_filters_lil_matrix = lil_matrix((dataset.shape[0], n_filters), dtype=np.int32)
    # for j in range(n_filters):
    #     if j % 1000 == 0:
    #         print(j)
    #     target = 1 / (j + 3)
    #     for i in range(num_points):
    #         if random.random() < target:
    #             data_filters_lil_matrix[i, j] = 1
    # print("nonzero:", data_filters_lil_matrix.count_nonzero())
    # data_filters_csc_matrix = csc_matrix(data_filters_lil_matrix)
    # pickle.dump(data_filters_csc_matrix, open("/home/zl20/composite-indexing-datasets/paper/paper/paper_query_attrs.pkl", "wb"))

    # One-hot attributes
    


if __name__ == "__main__":
    query_matrix = fvecs_read("~/bigann/biganntest/data/msong/msong_query.fvecs", np.int32)
    dataset = fvecs_read("~/bigann/biganntest/data/msong/msong_base.fvecs", np.int32)

    print(query_matrix.shape)
    print(dataset.shape)

    # 19 attrs, each with (i * 0.05) selectivity, 1 <= i <= 19
    data_filters_lil_matrix = lil_matrix((dataset.shape[0], 19), dtype=np.int32)
    random.seed(10)
    for i in range(19):
        target = (i + 1) * 0.05
        for j in range(dataset.shape[0]):
            if random.random() <= target:
                data_filters_lil_matrix[j, i] = 1
    pickle.dump(data_filters_lil_matrix, open("~/bigann/biganntest/data/msong/msong_data_attrs.pkl", "wb"))

    # 8 queries for each attr
    query_filters_lil_matrix = lil_matrix((query_matrix.shape[0], 19))
    for i in range(query_matrix.shape[0]):
        if i % 25 < 19:
            query_filters_lil_matrix[i, i % 25] = 1
    pickle.dump(query_filters_lil_matrix, open("~/bigann/biganntest/data/msong/msong_query_attrs.pkl", "wb"))

