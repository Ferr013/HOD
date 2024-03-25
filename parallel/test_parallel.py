import warnings
warnings.simplefilter("ignore")

import sys
import multiprocessing
from multiprocessing.shared_memory import SharedMemory
import uuid

import numpy as np
from scipy import integrate, special

def init(mem):
    global mem_id
    mem_id = mem
    return

def func(x, a):
    return a * special.j0(x)

def make_integral_array(args):
    _args_, j0zs, shape, nzeros_node, job_id = args
    shmem = SharedMemory(name=f'{mem_id}', create=False)
    intgr = np.ndarray(shape, buffer=shmem.buf, dtype=np.float64)
    start = job_id * nzeros_node
    end = start + nzeros_node
    intgr[start:end] = [integrate.quad(func, j0z[0], j0z[1], args = (_args_))[0]
                        for j0z in j0zs[start:end]]
    # print(f'Job ID: {job_id} | zeros node {nzeros_node} -> s: {start} | e: {end} \n'\
    #       ,'Bessel zeros : \n', j0zs[start:end]\
    #       ,'Integral values : \n', intgr[start:end])
    return

def make_integral_array_sharemem(_args_, nzeros, cores=None):
    if cores is None:
        cores = 6
        # cores = multiprocessing.cpu_count()
    print(f'# Cores: {cores}')
    # j0zs = np.append(1e-3, special.jn_zeros(0, nzeros))
    j0zs = np.append(1e-3,np.repeat(special.jn_zeros(0, nzeros),2)[:-1]).reshape(-1,2)
    args =  [(_args_, j0zs, nzeros, nzeros//int(cores), i) for i in range(int(cores))]
    exit = False
    try:
        global mem_id
        mem_id = str(uuid.uuid1())[:30] #Avoid OSError: [Errno 63]
        nbytes = nzeros * np.float64(1).nbytes
        shd_mem = SharedMemory(name=f'{mem_id}', create=True, size=nbytes)
        method = 'spawn'
        if sys.platform.startswith('linux'):
            method = 'fork'
        ctx = multiprocessing.get_context(method)
        pool = ctx.Pool(processes=cores, maxtasksperchild=1,
                        initializer=init, initargs=(mem_id,)
                        )
        try:
            pool.map_async(make_integral_array, args, chunksize=1).get(timeout=10_000)
        except KeyboardInterrupt:
            print("Caught kbd interrupt")
            pool.close()
            exit = True
        else:
            pool.close()
            pool.join()
            local_itgrd = np.ndarray(nzeros, buffer=shd_mem.buf, dtype=np.float64).copy()
    finally:
        shd_mem.close()
        shd_mem.unlink()
        if exit:
            sys.exit(1)
    return local_itgrd

if __name__ == "__main__":
    print('TEST CALC INTEGRAL')
    NCORES = None
    if len(sys.argv) > 1:
        print ('argument list', sys.argv)
        NCORES = int(sys.argv[1])
    NZEROS = 100_000
    mem_id = None
    _args_ = 1
    integral_parts = make_integral_array_sharemem(_args_, NZEROS, cores=NCORES)
    j_0_zeros = np.append(1e-3, special.jn_zeros(0, NZEROS))
    # print(j_0_zeros[:3], '...', j_0_zeros[-3:])
    # print(integral_parts[:3], '...', integral_parts[-3:])
    print(f'RESULT: {np.sum(integral_parts[:])}')
