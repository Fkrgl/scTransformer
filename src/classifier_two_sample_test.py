import numpy as np
import torch
from metrics import c2st
from sklearn.decomposition import PCA

def run_c2st(X, Y, clf):
    acc_XX = c2st(X, X, classifier=clf)
    acc_XY = c2st(X, Y, classifier=clf)
    return acc_XX, acc_XY

def get_projected_data(original, generated, n_components=50):
    pca = PCA(n_components=n_components)
    #pca.fit(np.vstack([original, generated]))
    original_U = pca.fit_transform(original)
    generated_U = pca.fit_transform(generated)
    # original_U = pca.transform(original)
    # generated_U = pca.transform(generated)
    original_U = torch.tensor(original_U).type(torch.DoubleTensor)
    generated_U = torch.tensor(generated_U).type(torch.DoubleTensor)
    return original_U, generated_U


path_original = '../data/pancreas_binned_original.npy'
path_generated = '../data/pancreas_binned_generated.npy'
original = np.load(path_original)
generated = np.load(path_generated)
original_U, generated_U = get_projected_data(original, generated)
original = torch.tensor(original).type(torch.DoubleTensor)
generated = torch.tensor(generated).type(torch.DoubleTensor)
original_U = torch.tensor(original_U).type(torch.DoubleTensor)
generated_U = torch.tensor(generated_U).type(torch.DoubleTensor)

X = torch.normal(2, 0.1,size=(1000, 200))
Y = torch.normal(0, 1, size=(1000, 200))



# test classiifer with random data
acc_XX, acc_XY = run_c2st(X, Y, 'rf')
print(f'normal distributed data using random forest\nacc_XX={acc_XX}\nacc_XY={acc_XY}')
acc_XX, acc_XY = run_c2st(X, Y, 'mlp')
print(f'normal distributed data using mlp\nacc_XX={acc_XX}\nacc_XY={acc_XY}')

# test classifier on scData
acc_XX, acc_XY = run_c2st(original, generated, 'rf')
print(f'sc Data using random forest\nacc_XX={acc_XX}\nacc_XY={acc_XY}')
acc_XX, acc_XY = run_c2st(original_U, generated_U, 'rf')
print(f'sc Data using random forest with PCA\nacc_XX={acc_XX}\nacc_XY={acc_XY}')

acc_XX, acc_XY = run_c2st(original, generated, 'mlp')
print(f'sc Data using mlp\nacc_XX={acc_XX}\nacc_XY={acc_XY}')
acc_XX, acc_XY = run_c2st(original_U, generated_U, 'mlp')
print(f'sc Data using mlp with PCA\nacc_XX={acc_XX}\nacc_XY={acc_XY}')

