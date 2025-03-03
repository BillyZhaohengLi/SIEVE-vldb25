import random
import math
import numpy as np
from scipy.sparse import csc_matrix, lil_matrix, csr_matrix
import pickle
import math
from heapq import heapify, heappush, heappop 
import os
import bisect

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
    query_matrix = fvecs_read("~/bigann/biganntest/data/sift2/sift_query.fvecs", np.int32)
    dataset = fvecs_read("~/bigann/biganntest/data/sift2/sift_base.fvecs", np.int32)

    print(query_matrix.shape)
    print(dataset.shape)

    # 2 attrs, each drawn from normal distribution
    np.random.seed(10)
    n_filters = 2
    data_filters_matrix = np.random.normal(loc=0, scale=1, size=(n_filters, dataset.shape[0]))
    data_filters_matrix.sort()
    data_filters_matrix = np.float32(data_filters_matrix)
    pickle.dump(np.transpose(data_filters_matrix), open("~/bigann/biganntest/data/sift2/sift_data_attrs.pkl", "wb"))

    # zipf distribution for queries
    random.seed(10)
    np.random.seed(10)
    sum = 10000
    q_list = []
    templates = {}
    zipf_coef = 200
    while len(q_list) < sum:
        i = 0
        # generate a query template
        template = [np.random.normal(loc = 0, scale = 1) for i in range(4)]
        template.sort()
        if random.random() > 0.5:
            tmp = template.pop(3)
            template.insert(1, tmp)
        else:
            tmp = template.pop(2)
            template.insert(1, tmp)
        if random.random() > 0.5:
            template = template[2:] + template[:2]
        template = tuple(template)
        # for j in range(4):
        #     template.append(np.random.normal(loc = 0, scale = 1))
        #     template.append(np.random.normal(loc = 0, scale = 1))
        #     template.append(np.random.normal(loc = 0, scale = 1))
        #     template.append(np.random.normal(loc = 0, scale = 1))
        # if template[0] > template[1]:
        #     template[0], template[1] = template[1], template[0]
        # if template[2] > template[3]:
        #     template[2], template[3] = template[3], template[2]
        # template = tuple(template)
        
        if template in templates:
            continue

        # template = random.sample(filter_list,20,weights=[1/(j+3) for j in range(200000)])
        while i < sum / zipf_coef / 5:
            i += 1
            q_list.append(template)
            if len(q_list) >= sum:
                break
        zipf_coef -= 1

        templates[template] = i
    print(zipf_coef)
    print(len(templates))

    sels = []
    sel = 0
    for k, v in templates.items():
        first_small = bisect.bisect_left(data_filters_matrix[0], k[0])
        first_large = bisect.bisect_right(data_filters_matrix[0], k[1])
        second_small = bisect.bisect_left(data_filters_matrix[1], k[2])
        second_large = bisect.bisect_right(data_filters_matrix[1], k[3])
        # print(first_small, first_large, second_small, second_large)
        # if max(first_small, second_small) <= min(first_large, second_large):
        #     val = (max(first_large, second_large) - min(first_small, second_small))
        # else:
        #     # If intervals are disjoint
        #     val = ((first_large - first_small) + (second_large - second_small))
        val = min(first_large, second_large) - max(first_small, second_small)
        sel += v * val
        for i in range(v):
            sels.append(val)
    print("selectivity:", sel / 10000)

    sels.sort()
    print(sels[0])
    print(sels[2000])
    print(sels[4000])
    print(sels[6000])
    print(sels[8000])
    print(sels[9999])

    tmp_matrix = []
    for k, v in templates.items():
        for i in range(v):
            tmp_matrix.append(list(k))

    # shuffle templates
    random.shuffle(tmp_matrix)
    random.shuffle(tmp_matrix)
    tmp_matrix = np.array(tmp_matrix, dtype = np.float32)
    print(tmp_matrix.shape)
    pickle.dump(tmp_matrix, open("~/bigann/biganntest/data/sift2/sift_query_attrs.pkl", "wb"))

    # # shuffle templates
    # attr_map = [i for i in range(200)]
    # random.shuffle(attr_map)

    # templates = []
    # for i in range(200):
    #     for j in range(scaled_template_probs[i]):
    #         templates.append(attr_map[i])
    # random.shuffle(templates)

    # print(attr_matrix.shape)
    # sel = 0
    # for i in range(len(templates)):
    #     sel += 1000000 * math.pow(1/(templates[i] + 3),0.5)
    # print("selectivity:", sel)

    # # to spmat
    # query_filters_lil_matrix = lil_matrix((10000, 200), dtype=np.int32)
    # for i in range(len(templates)):
    #     query_filters_lil_matrix[i,templates[i]] = 1
    
    # # print("query attrs shape:", query_attrs.shape)
    # pickle.dump(csr_matrix(query_filters_lil_matrix), open("~/bigann/biganntest/data/sift/sift_query_attrs.pkl", "wb"))
