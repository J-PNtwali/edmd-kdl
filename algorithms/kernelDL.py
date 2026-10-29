import numpy as np
import torch
import torch.nn as nn
import utils.kernels as kernels

class kernelDLBase:
    """
    Base class for EDMD with kernel-based dictionary learning for data-driven dynamical systems methods
    """
    def __init__(self, X, Y, Z, kernel, gamma):
        """
        Kernel dictionary learning initialization.

        Args:
            X (np.ndarray): training data matrix, shape (n, m), 
                where n is the number of features and m is the number of samples

            Y (np.ndarray): training data matrix after a fixed time lag, shape (n, m)

            Z (np.ndarray): matrix of collocation points, shape (n, m_c)

            kernel (object): kernel function object with:
                * a method `__call__` that evaluates the kernel function at given data points,
            
            gamma (float): nugget parameter for the first-level optimization
            
            beta (float): scaling parameter for the loss on Y
            
            delta (float): ridge regularization on the EDMD matrix K
        """
        self.X = X
        self.Y = Y
        self.Z = Z  # collocation points. set to np.column_stack([self.X, self.Y]) to match the derivation
        self.kernel = kernel
        self.gamma = gamma
        self.m = X.shape[1]  
        self.n = X.shape[0]  
        self.mc = self.Z.shape[1]   

        # calculate the Gram matrix G and the matrix Q = L^T(G + gamma * I)^{-1}
        self._assemble_G_Q_matrices_()
        
        # evaluate the embedding map at X and Y for use in optimization
        self.X_e = torch.from_numpy(self.embedding_map(self.X))
        self.Y_e = torch.from_numpy(self.embedding_map(self.Y))
    
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
    def __init__(self, X:np.ndarray, Y:np.ndarray, Z:np.ndarray, kernel:object, p:int, fixed_feature_fn:callable=None, 
                 decoder:bool=False, gamma:float=1e-12, beta1:float=1.0, 
                 beta2:float=0.0, beta3:float=0.0, delta:float=0.0, rcond:float=None, seed=42):
        """
        Kernel dictionary learning initialization.

        Args:
            X (np.ndarray): training data matrix, shape (n, m), 
                where n is the number of features and m is the number of samples

            Y (np.ndarray): training data matrix after a fixed time lag, shape (n, m)

            Z (np.ndarray): matrix of collocation points, shape (n, m_c)

            kernel (object): kernel function object with:
                * a method `__call__` that evaluates the kernel function at given data points,
            
            p (int, >= 1): number of dictionary features to learn

            fixed_feature_fn (callable): non-trainable dictionary map. Ignored if decoder is True. Default: None
            
            Z (np.ndarray): matrix of collocation points for the decoder, shape (n, m_tilde). Default: None

            decoder (bool, default False): whether or not to learn a decoder map from the latent space to the original state space. Default: False
            
            gamma (float): ridge parameter for kernel ridge regression (first-level optim)
            
            beta1 (float): weight for the EDMD loss

            beta2 (float): weight for the decoder loss

            beta3 (float): weight for the RKHS complexity penalty

            delta (float): ridge parameter for EDMD
        """
        super().__init__(X, Y, Z, kernel, gamma)
        self.p = p        
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.delta = delta
        self.fixed_feature_fn = fixed_feature_fn if not decoder else None
        self.rcond = rcond
        self.seed = seed

        self.decoder = decoder
        if self.decoder:
            self.Z_e = torch.from_numpy(self.embedding_map(self.Z)).contiguous()

        if self.fixed_feature_fn:
            self._X = torch.from_numpy(self.fixed_feature_fn(self.X)).contiguous()
            self._Y = torch.from_numpy(self.fixed_feature_fn(self.Y)).contiguous()

        # initialize pseudo values for the dictionary mapping (to be updated during training)
        self.U = nn.Parameter(torch.randn(self.p, self.mc, dtype=torch.float64, generator=torch.Generator().manual_seed(self.seed))) 

        # numpy to torch
        self.G_inv = torch.from_numpy(self.G_inv).contiguous()
        self.Z = torch.from_numpy(self.Z).contiguous()
        self.Q = torch.from_numpy(self.Q).contiguous()

    def loss_fn(self, vamp=False):
        r'''
        Compute the final objective function J(U) over the pseudo values for kernel dictionary learning:

            J(U) = || U Q1^T (I_{2m} -  \Phi_X^T (\Phi_X \Phi_X^T + \delta I_N))^{\dagger} \Phi_X ||_F^2 
                                            + \beta * ||Y(I_m - \Psi_X^T(\Psi_X \Psi_X^T + \delta I)^{\dagger} \Psi_X ||_F^2,

        where Q1 and Q2 are the two parts of the Q matrix split along columns, and ||.||_F denotes the Frobenius norm.
        '''
        # evaluate the dictionary at X and Y
        weights = self.G_inv @ self.U.T
        Psi_X = torch.cat((self._X, weights.T @ self.X_e)) if self.fixed_feature_fn else weights.T @ self.X_e
        Psi_Y = torch.cat((self._Y, weights.T @ self.Y_e)) if self.fixed_feature_fn else weights.T @ self.Y_e
        
        # calculate the EDMD matrix
        Psi_XX = Psi_X @ Psi_X.T
        Psi_YX = Psi_Y @ Psi_X.T
        #print(f"Psi_X: {Psi_X.shape}, Psi_XX: {Psi_XX.shape}, Psi_YX: {Psi_YX.shape}")
        K = torch.linalg.lstsq(Psi_XX.T + self.delta * torch.eye(Psi_XX.shape[0]), Psi_YX.T, rcond=self.rcond).solution.T

        # decoder matrix 
        if self.decoder:
            Psi_Z = weights.T @ self.Z_e
            Psi_ZZ = Psi_Z @ Psi_Z.T
            ZPsi_Z = self.Z @ Psi_Z.T  
            #print(f"Psi_Z: {Psi_Z.shape}, Psi_ZZ: {Psi_ZZ.shape}, ZPsi_Z: {ZPsi_Z.shape}")
            D = torch.linalg.lstsq(Psi_ZZ.T, ZPsi_Z.T, rcond=self.rcond).solution.T
        
        # EDMD loss
        loss1 = torch.norm(Psi_Y - K @ Psi_X, p='fro')**2
        if vamp:
            # squared eigenvals
            eigvals = torch.linalg.eigvals(K)
            vamp = -torch.sum(eigvals.real**2)
            imag_penalty = torch.sum(eigvals.imag**2)
            loss1 += 1e-6 * vamp + 0.0 * imag_penalty

        # reconstruction loss for the decoder
        loss2 = torch.norm(self.Z - D @ Psi_Z, p='fro')**2 if self.decoder else 0.0

        # complexity penalty (squared RKHS norm)
        loss3 = torch.norm(self.Q @ self.U.T)**2

        return self.beta1 * loss1 + self.beta2 *  loss2 + self.beta3 * loss3
    
    def vamp_loss_fn(self):
        # evaluate the dictionary at X and Y
        weights = torch.from_numpy(self.G_inv) @ self.U.T
        Psi_X = torch.cat((self._X, weights.T @ self.X_e)) if self.fixed_feature_fn else weights.T @ self.X_e
        Psi_Y = torch.cat((self._Y, weights.T @ self.Y_e)) if self.fixed_feature_fn else weights.T @ self.Y_e
        
        # covariance matrices
        Psi_XX = Psi_X @ Psi_X.T 
        Psi_YX = Psi_Y @ Psi_X.T 
        Psi_YY = Psi_Y @ Psi_Y.T 
        C00 = 0.5 * (Psi_XX + Psi_YY) 
        C01 = 0.5 * (Psi_YX + Psi_YX.T)

        # solve generalized eigenvalue problem
        U,S,Vt = torch.linalg.svd(C00)
        energy = torch.cumsum(S)/torch.sum(S)
        r = torch.argwhere(energy >= 0.995)[0].item()
        L = U[:,:r] / S[:r]
        R = L.T @ C01 @ L
        eigvals, eigvecs = torch.linalg.eig(R)
        eigvecs = L @ eigvecs

        # vamp-2 score loss
        vamp2 = torch.linalg.eigvals(K).real**2
        loss1 = - vamp2.sum()

        # complexity penalty (squared RKHS norm)
        loss2 = torch.norm(torch.from_numpy(self.Q) @ self.U.T)**2

        return loss1 + self.beta1 * loss2         
      
    def optimize(self, n_iters=10, beta1=None, beta2=None, beta3=None, delta=None, vamp=False, loss_hist=False, verbose=True, verbose_after=1, lr=1, max_iter=20, max_eval=None, tolerance_grad=1e-07,
                  tolerance_change=1e-09, history_size=100, line_search_fn="strong_wolfe"):
        #===== set hyperparameters
        if beta1 is not None:
            self.beta1 = beta1
        if beta2 is not None:
            self.beta2 = beta2
        if beta3 is not None:
            self.beta3 = beta3
        if delta is not None:
            self.delta = delta

        #====== L-BFGS optimizer 
        optimizer = torch.optim.LBFGS(
            [self.U],
            lr=lr,
            max_iter=max_iter,
            max_eval=max_eval,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            history_size=history_size,
            line_search_fn=line_search_fn,  
        )

        #====== Training loop
        def closure():
            optimizer.zero_grad()
            loss = self.loss_fn(vamp=vamp) #if not vamp else self.vamp_loss_fn()
            loss.backward()
            return loss         

        losses = []
        for step in range(n_iters):
            loss = optimizer.step(closure)   
            losses.append(loss.item())

            if verbose and step % verbose_after == 0:
                print(f"step {step:3d} | loss {loss.item():.1e}")
        
        if verbose:
            print(f"\nFinal loss : {losses[-1]:.1e}")
        
        #====== Assemble EDMD matrix and instantiate the learned dictionary map
        with torch.no_grad():
            self.weights = (self.G_inv @ self.U.T).numpy() 
            self._assemble_edmd_matrix_()
            self._edmd_spectral_decomposition_()

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
        return self.K @ z
    
    def decoder_fn(self, z):
        '''
        Evaluate the learned decoder map
        '''
        if not self.decoder:
            raise ValueError("Decoder is not enabled. Set decoder=True when initializing the EDMDkDL class.")
        return self.D @ z
        
    def _assemble_edmd_matrix_(self):
        with torch.no_grad():
            weights = self.G_inv @ self.U.T
            Psi_X = torch.cat((self._X, weights.T @ self.X_e)) if self.fixed_feature_fn else weights.T @ self.X_e
            Psi_Y = torch.cat((self._Y, weights.T @ self.Y_e)) if self.fixed_feature_fn else weights.T @ self.Y_e

            self.K = torch.linalg.lstsq(Psi_X.T, Psi_Y.T, rcond=self.rcond).solution.T.numpy()

            # decoder matrix
            if self.decoder:
                Psi_Z = weights.T @ self.Z_e
                self.D = torch.linalg.lstsq(Psi_Z.T, self.Z.T, rcond=self.rcond).solution.T.numpy()

    def _edmd_spectral_decomposition_(self):
        '''
        Estimate Koopman eigenfunctions and associated eigenvalues 
        '''
        # eigendecomposition
        eigvals, eigvecs = np.linalg.eig(self.K.T)
        # sort by decreasing order
        idx = np.argsort(eigvals)[::-1]
        self.eigvals, self.eigvecs = eigvals[idx], eigvecs[idx]
       



     

class EDMDkDLv2(kernelDLBase):
    '''
    Extended dynamic mode decomposition with kernel dictionary learning (EDMD-kDL) class, inherits from kernelDLBase.
    '''
    def __init__(self, X:np.ndarray, Y:np.ndarray, Z:np.ndarray, kernel:object, p:int, kernel_dec:object=None, fixed_feature_fn:callable=None,
            decoder:bool=False, gamma1:float=1e-14, gamma2:float=1e-14, beta1:float=1.0, beta2:float=0.0, beta3:float=0.0, delta:float=0.0, rcond:float=None, seed=42):
        """
        Kernel dictionary learning initialization.

        Args:
            X (np.ndarray): training data matrix, shape (n, m), 
                where n is the number of features and m is the number of samples

            Y (np.ndarray): training data matrix after a fixed time lag, shape (n, m)

            Z (np.ndarray): matrix of collocation points, shape (n, m_c)

            kernel (object): kernel function object with:
                * a method `__call__` that evaluates the kernel function at given data points,
            
            p (int, >= 1): number of dictionary features to learn

            kernel_dec (object): kernel function object for the decoder. Ignored if decoder is False. Default: None

            fixed_feature_fn (callable): non-trainable dictionary map. Ignored if decoder is True. Default: None
            
            decoder (bool, default False): whether or not to learn a decoder map from the latent space to the original state space. Default: False
            
            gamma1 (float): ridge parameter for the encoder (first-level optimization)

            gamma2 (float): ridge parameter for the decoder 

            beta1 (float): weight of the EDMD loss

            beta2 (float): weight of the reconstruction (decoder) loss

            beta3 (float): weight of the regularization term (squared RKHS norm of the learned dictionary)

            delta (float): ridge parameter for EDMD
        """
        super().__init__(X, Y, Z, kernel, gamma1)
        self.p = p        
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.delta = delta
        self.rcond = rcond
        self.seed = seed
        
        self.decoder = decoder
        self.fixed_feature_fn = fixed_feature_fn if not decoder else None
        if self.decoder:
            self.kernel_dec = kernel_dec if kernel_dec is not None else self.kernel
            self.Z_e = torch.from_numpy(self.embedding_map(self.Z))

        if self.fixed_feature_fn:
            self._X = torch.from_numpy(self.fixed_feature_fn(self.X))
            self._Y = torch.from_numpy(self.fixed_feature_fn(self.Y))

        # initialize pseudo values for the dictionary mapping (to be updated during training)
        self.U = nn.Parameter(torch.randn(self.p, self.mc, dtype=torch.float64, generator=torch.Generator().manual_seed(self.seed))) 
         
    def loss_fn(self, vamp=False):
        r'''
        Compute the final objective function J(U) over the pseudo values for kernel dictionary learning:

            J(U) = || U Q1^T (I_{2m} -  \Phi_X^T (\Phi_X \Phi_X^T + \delta I_N))^{\dagger} \Phi_X ||_F^2 
                                            + \beta * ||Y(I_m - \Psi_X^T(\Psi_X \Psi_X^T + \delta I)^{\dagger} \Psi_X ||_F^2,

        where Q1 and Q2 are the two parts of the Q matrix split along columns, and ||.||_F denotes the Frobenius norm.
        '''
        # evaluate the dictionary at X and Y
        weights = torch.from_numpy(self.G_inv) @ self.U.T
        Psi_X = torch.cat((self._X, weights.T @ self.X_e)) if self.fixed_feature_fn else weights.T @ self.X_e
        Psi_Y = torch.cat((self._Y, weights.T @ self.Y_e)) if self.fixed_feature_fn else weights.T @ self.Y_e
        
        # calculate the EDMD matrix
        Psi_XX = Psi_X @ Psi_X.T
        Psi_YX = Psi_Y @ Psi_X.T
        K = torch.linalg.lstsq(Psi_XX.T + self.delta * torch.eye(Psi_XX.shape[0]), Psi_YX.T, rcond=self.rcond).solution.T

        # decoder matrix 
        if self.decoder:
            Psi_Z = weights.T @ self.Z_e
            G2 = kernels.gramian(Psi_Z, self.kernel_dec)
            V = torch.linalg.lstsq(G2 + self.gamma2 * torch.eye(G2.shape[0]), torch.from_numpy(self.Z).T, rcond=self.rcond).solution.T
        
        # EDMD loss
        loss1 = torch.norm(Psi_Y - K @ Psi_X, p='fro')**2
        if vamp:
            # squared eigenvals
            eigvals = torch.linalg.eigvals(K)
            vamp = -torch.sum(eigvals.real**2)
            imag_penalty = torch.sum(eigvals.imag**2)
            loss1 += 1e-6 * vamp + 0.0 * imag_penalty

        # reconstruction loss for the decoder
        loss2 = torch.norm(V, p='fro')**2 if self.decoder else 0.0

        # complexity penalty (squared RKHS norm)
        loss3 = torch.norm(torch.from_numpy(self.Q) @ self.U.T)**2

        return self.beta1*loss1 + self.beta2*loss2 + self.beta3*loss3
      
    def optimize(self, n_iters=10, beta1=None, beta2=None, delta=None, vamp=False, loss_hist=False, verbose=True, verbose_after=1, lr=1, max_iter=20, max_eval=None, tolerance_grad=1e-07,
                  tolerance_change=1e-09, history_size=100, line_search_fn="strong_wolfe"):
        #===== set hyperparameters
        if beta1 is not None:
            self.beta1 = beta1
        if beta2 is not None:
            self.beta2 = beta2
        if delta is not None:
            self.delta = delta

        #====== L-BFGS optimizer 
        optimizer = torch.optim.LBFGS(
            [self.U],
            lr=lr,
            max_iter=max_iter,
            max_eval=max_eval,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            history_size=history_size,
            line_search_fn=line_search_fn,  
        )

        #====== Training loop
        def closure():
            optimizer.zero_grad()
            loss = self.loss_fn(vamp=vamp) #if not vamp else self.vamp_loss_fn()
            loss.backward()
            return loss         

        losses = []
        for step in range(n_iters):
            loss = optimizer.step(closure)   
            losses.append(loss.item())

            if verbose and step % verbose_after == 0:
                print(f"step {step:3d} | loss {loss.item():.1e}")

        print(f"\nFinal loss : {losses[-1]:.1e}")
        
        #====== Assemble EDMD matrix and instantiate the learned dictionary map
        with torch.no_grad():
            self.weights = (torch.from_numpy(self.G_inv) @ self.U.T).numpy() 
            self._assemble_edmd_matrix_()
            self._edmd_spectral_decomposition_()

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
        return self.K @ z
    
    def decoder_fn(self, z):
        '''
        Evaluate the learned decoder map
        '''
        if not self.decoder:
            raise ValueError("Decoder is not enabled. Set decoder=True when initializing the EDMDkDL class.")
        return self.D @ z
        
    def _assemble_edmd_matrix_(self):
        with torch.no_grad():
            weights = torch.from_numpy(self.G_inv) @ self.U.T
            Psi_X = torch.cat((self._X, weights.T @ self.X_e)) if self.fixed_feature_fn else weights.T @ self.X_e
            Psi_Y = torch.cat((self._Y, weights.T @ self.Y_e)) if self.fixed_feature_fn else weights.T @ self.Y_e

            self.K = torch.linalg.lstsq(Psi_X.T, Psi_Y.T, rcond=self.rcond).solution.T.numpy()

            # decoder matrix
            if self.decoder:
                Psi_Z = weights.T @ self.Z_e
                self.D = torch.linalg.lstsq(Psi_Z.T, torch.from_numpy(self.Z).T, rcond=self.rcond).solution.T.numpy()

    def _edmd_spectral_decomposition_(self):
        '''
        Estimate Koopman eigenfunctions and associated eigenvalues 
        '''
        # eigendecomposition
        eigvals, eigvecs = np.linalg.eig(self.K.T)
        # sort by decreasing order
        idx = np.argsort(eigvals)[::-1]
        self.eigvals, self.eigvecs = eigvals[idx], eigvecs[idx]
        



               

        
    
    

