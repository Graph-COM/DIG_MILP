import numpy as np
import scipy
import argparse
from tqdm import tqdm
import random

def generate_setcover(nrows, ncols, density_list, filename, rng, max_coef=100):
    """
    Generates a setcover instance with specified characteristics, and writes
    it to a file in the LP format.

    Approach described in:
    E.Balas and A.Ho, Set covering algorithms using cutting planes, heuristics,
    and subgradient optimization: A computational study, Mathematical
    Programming, 12 (1980), 37-60.

    Parameters
    ----------
    nrows : int
        Desired number of rows
    ncols : int
        Desired number of columns
    density: float between 0 (excluded) and 1 (included)
        Desired density of the constraint matrix
    filename: str
        File to which the LP will be written
    rng: numpy.random.RandomState
        Random number generator
    max_coef: int
        Maximum objective coefficient (>=1)
    """
    density = random.choice(density_list)
    nnzrs = int(nrows * ncols * density)

    assert nnzrs >= nrows  # at least 1 col per row
    assert nnzrs >= 2 * ncols  # at leats 2 rows per col

    # compute number of rows per column
    indices = rng.choice(ncols, size=nnzrs)  # random column indexes
    indices[:2 * ncols] = np.repeat(np.arange(ncols), 2)  # force at leats 2 rows per col
    _, col_nrows = np.unique(indices, return_counts=True)

    # for each column, sample random rows
    indices[:nrows] = rng.permutation(nrows) # force at least 1 column per row
    i = 0
    indptr = [0]
    for n in col_nrows:

        # empty column, fill with random rows
        if i >= nrows:
            indices[i:i+n] = rng.choice(nrows, size=n, replace=False)

        # partially filled column, complete with random rows among remaining ones
        elif i + n > nrows:
            remaining_rows = np.setdiff1d(np.arange(nrows), indices[i:nrows], assume_unique=True)
            indices[nrows:i+n] = rng.choice(remaining_rows, size=i+n-nrows, replace=False)

        i += n
        indptr.append(i)

    # objective coefficients
    c = rng.randint(max_coef, size=ncols) + 1

    # sparce CSC to sparse CSR matrix
    A = scipy.sparse.csc_matrix(
            (np.ones(len(indices), dtype=int), indices, indptr),
            shape=(nrows, ncols)).tocsr()
    indices = A.indices
    indptr = A.indptr

    # write problem
    with open(filename, 'w') as file:
        file.write("maximize\nOBJ:")
        file.write("".join([f" -{c[j]} x{j+1}" for j in range(ncols)]))
        file.write("\n\nsubject to\n")
        for i in range(nrows):
            row_cols_str = "".join([f" -1 x{j+1}" for j in indices[indptr[i]:indptr[i+1]]])
            file.write(f"C{i+1}:" + row_cols_str + f" <= -1\n")
        file.write("\nbinary\n")
        file.write("".join([f" x{j+1}" for j in range(ncols)]))

def main():
    parser = argparse.ArgumentParser(description='LP file parser')
    parser.add_argument('--primal_folder', dest = 'primal_folder', type = str, default = './testset/m200n400/', help = 'which folder to get the lp instances')
    parser.add_argument('--num_instance', dest = 'num_instance', type = int, default = 500, help = 'the number of instances')
    parser.add_argument('--nrows', dest = 'nrows', type = int, default = 200, help = 'number of rows')
    parser.add_argument('--ncols', dest = 'ncols', type = int, default = 400, help = 'number of columns')
    parser.add_argument('--density', dest = 'density', type = float, default = [0.15, 0.20, 0.25, 0.30,0.35], help = 'density')
    parser.add_argument('--seed', dest = 'seed', type = int, default = 123, help = 'random seed')
    parser.add_argument('--max_coeff', dest = 'max_coeff', type = int, default = 100, help = 'maximum coefficient of objective')
    args = parser.parse_args()

    for instance_idx in tqdm(range(args.num_instance)):
        lp_file = args.primal_folder+'val/'+str(instance_idx)+'.lp'
        random_seed = random.randint(0, 999999)
        rng = np.random.RandomState(random_seed)
        generate_setcover(nrows = args.nrows, ncols = args.ncols, density_list = args.density, filename = lp_file, rng = rng, max_coef = args.max_coeff)

if __name__ == '__main__':
    main()