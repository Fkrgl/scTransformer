import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from metrics import c2st

def euclidean_distance_centroids(real: torch.tensor, generated: torch.tensor):
    '''
    function calculates the centroids of each data set. The centroid is obtained by the mean along the gene axis over
    all samples. Returns the mean distance between centroids
    '''
    centroid_real = torch.mean(real, dim=1)
    centroid_gen = torch.mean(generated, dim=1)
    print(centroid_real.shape)
    dist = np.linalg.norm(centroid_real - centroid_gen)
    return dist

# load tensors
path_original = '../data/pancreas_binned_original.npy'
path_generated = '../data/pancreas_binned_generated.npy'
original = np.load(path_original)
generated = np.load(path_generated)
original = torch.tensor(original).type(torch.DoubleTensor)
generated = torch.tensor(generated).type(torch.DoubleTensor)
# original = StandardScaler().fit_transform(original)
# generated = StandardScaler().fit_transform(generated)
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

original, generated = get_projected_data(original, generated)
def create_artificial_scData(n_samples, n_genes, n_bins, zero_prob):
    shape = (n_samples, n_genes)
    probability_matrix = torch.full(shape, zero_prob)
    mask = torch.bernoulli(probability_matrix).bool()
    random_ints = torch.tensor(np.random.choice(np.arange(n_bins), size=shape))
    random_ints[mask] = 0
    return random_ints

# X = create_artificial_scData(1000, 200, 100, 0.9)
# X = StandardScaler().fit_transform(X)
# X = torch.tensor(X).type(torch.DoubleTensor)
# X = np.random.choice(np.arange(100), size=(1000,200)).astype(float)
# Y = np.random.choice(np.arange(100, 200), size=(1000,200)).astype(float)
# noise = 0.1 * np.random.randn(1000, 200)
# X += noise
# X = StandardScaler().fit_transform(X)
# print(X)

# X = torch.tensor(X).type(torch.DoubleTensor)
# Y = torch.tensor(Y).type(torch.DoubleTensor)
# X = torch.normal(2, 0.1,size=(1000, 200))
# Y = torch.normal(0, 1, size=(1000, 200))
# accuracy1 = c2st(X, X, classifier='rf')
# accuracy2 = c2st(X, Y, classifier='rf')
# print(f'accuracy X, X: {accuracy1}')
# print(f'accuracy X, Y: {accuracy2}')

# https://www.kaggle.com/code/onurtunali/maximum-mean-discrepancy
def MMD(x, y, kernel):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx  # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy  # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz  # Used for C in (1)

    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))

    if kernel == "multiscale":

        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a ** 2 * (a ** 2 + dxx) ** -1
            YY += a ** 2 * (a ** 2 + dyy) ** -1
            XY += a ** 2 * (a ** 2 + dxy) ** -1

    if kernel == "rbf":

        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5 * dxx / a)
            YY += torch.exp(-0.5 * dyy / a)
            XY += torch.exp(-0.5 * dxy / a)

    return torch.mean(XX + YY - 2. * XY)

X = np.random.choice(np.arange(100), size=(1000,200)).astype(float)
Y = np.random.choice(np.arange(100, 200), size=(1000,200)).astype(float)
X = torch.tensor(X).type(torch.DoubleTensor)
Y = torch.tensor(Y).type(torch.DoubleTensor)
print(MMD(original, generated, 'rbf'))
print(MMD(original, original, 'rbf'))

print(euclidean_distance_centroids(original, generated))
print(euclidean_distance_centroids(original, original))