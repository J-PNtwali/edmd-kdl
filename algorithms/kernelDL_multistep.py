import numpy as np
import torch
import torch.nn as nn
import utils.kernels as kernels
from utils.helpers import get_device, set_seed

device = get_device()


class kernelDLBase:
    """
    Base class for EDMD with kernel-based dictionary learning for data-driven dynamical systems methods
    """
    def __init__(self, Z, kernel, gamma):
        """
        Kernel dictionary learning initialization.

        Args:
    
            Z (np.ndarray): matrix of collocation points, shape (n, m_c)

            kernel (object): kernel function object with:
                * a method `__call__` that evaluates the kernel function at given data points,
            
            gamma (float): nugget parameter for the first-level optimization
            
            beta (float): scaling parameter for the loss on Y
            
            delta (float): ridge regularization on the EDMD matrix K
        """
        self.Z = Z  # collocation points. set to np.column_stack([self.X, self.Y]) to match the derivation
        self.kernel = kernel
        self.gamma = gamma
        self.mc = self.Z.shape[1]   

        # calculate the Gram matrix G and the matrix Q = L^T(G + gamma * I)^{-1}
        self._assemble_G_Q_matrices_()
    
    def embedding_map(self, x):
        r"""
        Embedding map (\xi) at a given data point x:
           x -->  k(Xc, x) 
        """        
        if x.ndim == 1:
            x = x[:, np.newaxis] 

        return  kernels.gramian2(self.Z, x, self.kernel)

    def _assemble_G_Q_matrices_(self):
        """
        Compute the mixed Gramian matrix at the training data points G.

        And calculate the inverse of G + gamma * I
        """
        self.G = kernels.gramian(self.Z, self.kernel)
        # singular value decomposition
        U, S, Ut = np.linalg.svd(self.G, hermitian=True)
        self.G_inv = (U / (S + self.gamma)) @ Ut   #np.linalg.pinv(self.G + self.gamma * np.eye(self.mc))
        L = U * np.sqrt(S)
        self.Q = L.T @ self.G_inv
        self.Qs = self.G @ self.G_inv
    

class EDMDkDL(kernelDLBase):
    '''
    Extended dynamic mode decomposition with kernel dictionary learning (EDMD-kDL) class, inherits from kernelDLBase.
    '''
    def __init__(self, X:np.ndarray, Y:np.ndarray, Z:np.ndarray, kernel:object, p:int,  fixed_feature_fn:callable=None, 
                 decoder:bool=False, gamma:float=1e-12, varproj:bool=False, rcond:float=None, seed=42):
        """
        Kernel dictionary learning initialization.

        Args:
            X (np.ndarray): training data matrix, shape (num_trajectories, n, m), 
                where n is the number of features and m is the number of samples

            Y (np.ndarray): training data matrix after a fixed time lag, shape (num_trajectories, n, m)

            Z (np.ndarray): matrix of collocation points, shape (n, m_c)

            kernel (object): kernel function object with:
                * a method `__call__` that evaluates the kernel function at given data points,
            
            p (int, >= 1): number of dictionary elements to learn

            fixed_feature_fn (callable): non-trainable dictionary map. Ignored if decoder is True. Default: None
            
            decoder (bool, default False): whether or not to learn a decoder map from the latent space to the original state space. Default: False
            
            gamma (float): ridge parameter for kernel ridge regression (first-level optim)
            
        """
        super().__init__(Z, kernel, gamma)
        if X.ndim == 2:
            X, Y = X[np.newaxis, :, :], Y[np.newaxis, :, :]
        self.X, self.Y = X, Y
        self.num_traj, self.n, self.m = X.shape

        self.p = p        
        self.fixed_feature_fn = fixed_feature_fn if not decoder else None
        self.varproj = varproj
        self.rcond = rcond
        self.decoder = decoder
        self.seed = seed
        set_seed(seed)

        #=== evaluate the embedding map at X and Y for use in optimization
        self.X_e = torch.from_numpy(
            np.concatenate([self.embedding_map(self.X[i, :, :])[np.newaxis,...] for i in range(self.num_traj)])
            )
        self.Y_e = torch.from_numpy(
            np.concatenate([self.embedding_map(self.Y[i, :, :])[np.newaxis,...] for i in range(self.num_traj)])
            )

        #=== evaluate fixed elements
        if self.decoder:
            self.Z_e = torch.from_numpy(self.embedding_map(self.Z))

        if self.fixed_feature_fn:
            self._X = torch.from_numpy(
                np.concatenate([self.fixed_feature_fn(self.X[i, :, :])[np.newaxis,...] for i in range(self.num_traj)])
            ).contiguous()
            self._Y = torch.from_numpy(
                np.concatenate([self.fixed_feature_fn(self.Y[i, :, :])[np.newaxis,...] for i in range(self.num_traj)])
            ).contiguous()

        self.N = self.p if decoder else self.p + (self._X.shape[1] if self.fixed_feature_fn else None)


        #=== initialize trainable parameters
        # dictionary matrix (pseudo values)
        self.U = nn.Parameter(torch.randn(self.p, self.mc, dtype=torch.float64, device=device))
        # edmd matrix
        self.K = nn.Parameter(torch.randn(self.N, self.N, dtype=torch.float64, device=device))  
        # decoder matrix
        if not self.varproj:
            self.D = nn.Parameter(torch.randn(self.n, self.N, dtype=torch.float64, device=device))

        # numpy to torch
        self.G_inv = torch.from_numpy(self.G_inv).contiguous()
        self.Z = torch.from_numpy(self.Z).contiguous()
        self.Q = torch.from_numpy(self.Q).contiguous()
         
    def loss_fn(self):
        r'''
        Compute the final objective function J(U) over the pseudo values for kernel dictionary learning:
        '''
        mse_loss = nn.MSELoss(reduction='mean')
        # evaluate the dictionary at X and Y
        weights = self.G_inv.to(device) @ self.U.T
        Psi_tilde_X = (self.X_e.to(device).transpose(1,2) @ weights.to(device)).transpose(1,2)
        Psi_tilde_Y = (self.Y_e.to(device).transpose(1,2) @ weights.to(device)).transpose(1,2)
        Psi_X = torch.cat((self._X.to(device), Psi_tilde_X), dim=1) if self.fixed_feature_fn else Psi_tilde_X
        Psi_Y = torch.cat((self._Y.to(device), Psi_tilde_Y), dim=1) if self.fixed_feature_fn else Psi_tilde_Y

        # decoder matrix 
        if self.decoder:
            Psi_Z = weights.T @ self.Z_e.to(device)
            if self.varproj:
                Psi_ZZ = Psi_Z @ Psi_Z.T
                ZPsi_Z = self.Z.to(device) @ Psi_Z.T  
                D = torch.linalg.lstsq(Psi_ZZ.T, ZPsi_Z.T, rcond=self.rcond).solution.T
        
        # EDMD loss
        loss1 = 0.0
        # for traj in range(self.num_traj):
        #     PsiX_traj = Psi_X[traj, :, :]
        #     PsiY_traj = Psi_Y[traj, :, :]

        #     for step in range(self.steps):
        #         Psi_Y_pred = self.K @ PsiX_traj if step == 0 else self.K @ Psi_Y_pred
        #         loss1 += mse_loss(PsiY_traj[:,step:], Psi_Y_pred[:,:(self.m-step)])  
        for step in range(self.steps):
            Psi_Y_pred = self.K @ Psi_X if step == 0 else self.K @ Psi_Y_pred
            loss1 += mse_loss(Psi_Y[:, :, step:], Psi_Y_pred[:, :, :(self.m-step)])

        # reconstruction loss for the decoder
        if self.varproj:
            loss2 = mse_loss(self.Z.to(device), D @ Psi_Z) if self.decoder else 0.0
        else:
            loss2 = mse_loss(self.Z.to(device), self.D @ Psi_Z) if self.decoder else 0.0

        # complexity penalty (squared RKHS norm)
        WW = self.Q.to(device) @ self.U.T
        loss3 = mse_loss(WW, torch.zeros_like(WW))  

        return self.beta1 * loss1 + self.beta2 *  loss2 + self.beta3 * loss3
    
    def optimize(self, n_iters=10, steps=1, beta1=1., beta2=1., beta3=1e-8, delta=None,
             loss_hist=False, verbose=True, verbose_after=1, optimizer='lbfgs', **kwargs):
        #===== set hyperparameters
        if steps is not None:
            self.steps = steps
        if beta1 is not None:
            self.beta1 = beta1
        if beta2 is not None:
            self.beta2 = beta2
        if beta3 is not None:
            self.beta3 = beta3
        if delta is not None:
            self.delta = delta

        #====== resolve optimizer class
        optimizer_map = {
            'lbfgs': torch.optim.LBFGS,
            'adam': torch.optim.Adam,
            'adamw': torch.optim.AdamW,
            'sgd': torch.optim.SGD,
            'rmsprop': torch.optim.RMSprop,
            'adagrad': torch.optim.Adagrad,
        }

        if isinstance(optimizer, str):
            key = optimizer.lower()
            if key not in optimizer_map:
                raise ValueError(
                    f"Unknown optimizer '{optimizer}'. "
                    f"Choose from {list(optimizer_map)} or pass an optimizer class directly."
                )
            optimizer_cls = optimizer_map[key]
        elif isinstance(optimizer, type) and issubclass(optimizer, torch.optim.Optimizer):
            optimizer_cls = optimizer
        else:
            raise TypeError(
                "optimizer must be a string (e.g. 'lbfgs', 'adam') or a torch.optim.Optimizer subclass."
            )

        opt = optimizer_cls([self.U, self.K], **kwargs) if self.varproj else optimizer_cls([self.U, self.K, self.D], **kwargs)

        #====== Training loop
        def closure():
            opt.zero_grad()
            loss = self.loss_fn()  # if not vamp else self.vamp_loss_fn()
            loss.backward()
            return loss

        losses = []
        for step in range(n_iters):
            loss = opt.step(closure)
            losses.append(loss.item())

            if verbose and step % verbose_after == 0:
                print(f"step {step:3d} | loss {loss.item():.1e}")

        if verbose:
            print(f"\nFinal loss : {losses[-1]:.1e}")

        #====== Assemble EDMD matrix and instantiate the learned dictionary map
        with torch.no_grad():
            self._edmd_spectral_decomposition_()
            weights = self.G_inv.to(device) @ self.U.T
            # decoder matrix
            if self.decoder and self.varproj:
                Psi_Z = weights.T @ self.Z_e.to(device)
                Psi_ZZ = Psi_Z @ Psi_Z.T
                ZPsi_Z = self.Z.to(device) @ Psi_Z.T  
                self.D = torch.linalg.lstsq(Psi_ZZ.T, ZPsi_Z.T, rcond=self.rcond).solution.T.cpu().numpy()

            self.weights = weights.cpu().numpy()

        #====== output loss history
        if loss_hist:
            return np.array(losses)
    
    def psi_tilde_fn(self, x):
        '''
        Learned complementary dictionary functions
        '''
        return self.weights.T @ self.embedding_map(x)
    
    def psi_fn(self,x):
        '''
        Full dictionary map
        '''
        if x.ndim == 1:
            x = x[:, np.newaxis]
        psi_x =  self.psi_tilde_fn(x)
        if self.fixed_feature_fn:
            psi_x = np.vstack([self.fixed_feature_fn(x), psi_x]) 
        return psi_x
    
    def eigen_fn(self,x):
        '''
        Approximated eigenfunctions of the Koopman operator
        '''
        return self.eigvecs.T @ self.psi_fn(x)
    
    def learned_linear_model(self, z):
        '''
        Evaluate the learned latent linear model
        '''
        with torch.no_grad():
            z1 = self.K.cpu() @ z
        return z1
    
    def decoder_fn(self, z):
        '''
        Evaluate the learned decoder map
        '''
        if not self.decoder:
            raise ValueError("Decoder is not enabled. Set decoder=True when initializing the EDMDkDL class.")
        with torch.no_grad():
            x = self.D.cpu() @ z
        return x

    def _edmd_spectral_decomposition_(self):
        '''
        Estimate Koopman eigenfunctions and associated eigenvalues 
        '''
        with torch.no_grad():
            # eigendecomposition
            eigvals, eigvecs = np.linalg.eig(self.K.detach().cpu().numpy().T)
            # sort by decreasing order
            idx = np.argsort(eigvals)[::-1]
            self.eigvals, self.eigvecs = eigvals[idx], eigvecs[idx]