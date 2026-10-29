import numpy as np
import utils.kernels as kernels

class eDMD():
    def __init__(self, X:np.array, Y:np.array, psi_fn:callable=None):
        self.X = X
        self.Y = Y
        self.psi_fn = psi_fn
    
    def train(self, rcond=None, dual=False):
        psiX = self.psi_fn(self.X) if self.psi_fn else self.X
        psiY = self.psi_fn(self.Y) if self.psi_fn else self.Y
        if not dual:
            self.K = np.linalg.lstsq(psiX.T.astype(np.float32), psiY.T.astype(np.float32), rcond=rcond)[0].T
        else:
            print(psiX.shape)
            self.K = psiY @ np.linalg.pinv(psiX.T @ psiX) @ psiX.T
    
    def linear_model(self, z):
        return self.K @ z


class KernelEDMD():
    def __init__(self, X:np.array, Y:np.array, kernel:object, rcond=None):
        self.X = X
        self.Y = Y
        self.kernel = kernel
        self.rcond= rcond

    def train(self):
        # assemble Gram matrices
        self.G = kernels.gramian(self.X, self.kernel)
        self.A = kernels.gramian2(self.X, self.Y, self.kernel)

        # construct K matrix
        Sigma2, Q = np.linalg.eigh(self.G)

        Sigma_pinv = np.array([1/np.sqrt(Sigma2[i]) if Sigma2[i] > 0 else 0 for i  in range(len(Sigma2))])
        QSigma_pinv = Q * Sigma_pinv

        self.K = QSigma_pinv.T @ self.A @ QSigma_pinv 

        # eigendecomposition
        self.eig_vals, self.eig_vectors = np.linalg.eig(self.K)

        # approximate Koopman eigenfunctions
        self.weights = QSigma_pinv @ self.eig_vectors
        self.eig_fn = lambda x: kernels.gramian2(self.X, x, self.kernel).T @ self.weights

        # approximate Koopman modes
        self.modes = np.linalg.lstsq(self.eig_fn(self.X).T, self.X.T, rcond=self.rcond)[0].T
        self.fitted_ = True

    def predict(self, x):
        if x.ndim ==1:
            x = x[:, np.newaxis]
        z = self.eig_vals * self.eig_fn(x).squeeze()
        return self.modes @ z



class rffEDMD():
    def __init__(self, X:np.array, Y:np.array, sigma:float, decoder=True, seed:int=42, rcond=None):
        self.X = X
        self.Y = Y
        self.sigma = sigma
        self.decoder = decoder
        self.seed = seed
        self.rcond= rcond

    def train(self, n_features):
        self.dim = self.X.shape[0]
        rng = np.random.RandomState(self.seed)

        # setup RFF map
        self.W = rng.normal(scale=1/self.sigma, size=(self.dim, n_features))
        self.b = rng.uniform(0, 2 * np.pi, size=n_features)
        self.psi_fn = lambda x: np.sqrt(2) * np.cos(x.T @ self.W + self.b).T if self.decoder else np.column_stack((x.T,  np.sqrt(2) * np.cos(x.T @ self.W + self.b))).T

        # construct K matrix
        psiX = self.psi_fn(self.X) 
        psiY = self.psi_fn(self.Y)
        self.K = np.linalg.lstsq(psiX.T, psiY.T, rcond=self.rcond)[0].T

        # decoder
        if self.decoder:
            self.D = np.linalg.lstsq(psiX.T, self.X.T, rcond=self.rcond)[0].T
            self.decoder_fn = lambda z: self.D @ z
        else:
            self.decoder_fn = lambda z: z[:self.dim,...] 
        
        # approximate Koopman modes
        self.fitted_ = True

    def predict(self, x):
        if x.ndim ==1:
            x = x[:, np.newaxis]
        z = self.psi_fn(x)
        z1 = self.K @ z
        return self.decoder_fn(z1)
