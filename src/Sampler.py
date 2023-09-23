import scanpy as scp
import numpy as np
import argparse

class Sampler:
    def __int__(self):
        self.data = None

    def read(self, in_path: str):
        '''
        read in h5ad file
        '''
        self.data = scp.read_h5ad(in_path)

    def write(self, adata, out_path: str):
        adata.write(out_path)

    def sample(self, n_sample: int):
        np.random.seed(42)
        if n_sample >= self.data.n_obs:
            return

        # Generate random indices for subsampling without replacement
        random_indices = np.random.choice(self.data.n_obs, n_sample, replace=False)

        # Subsample the AnnData object based on random indices
        subsampled_data = self.data[random_indices, :]
        return subsampled_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='subsamples a adata object')
    parser.add_argument('in_path', type=str)
    parser.add_argument('out_path', type=str)
    parser.add_argument('n_sample', type=int)
    args = parser.parse_args()

    sampler = Sampler()
    sampler.read(args.in_path)
    sub_data = sampler.sample(args.n_sample)
    print(f'shape of data: {sub_data}')
    sampler.write(sub_data, args.out_path)
